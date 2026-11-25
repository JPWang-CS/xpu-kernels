#ifndef XPU_RMS_NORM_PREFETCH_ROW_RETAIN_KERNEL_H_
#define XPU_RMS_NORM_PREFETCH_ROW_RETAIN_KERNEL_H_

#include <kernel_operator.h>
#include <kernel_tiling/kernel_tiling.h>
#include <xpu_kernels/rms_norm/regbase.hpp>
#include <xpu_kernels/rms_norm/reduce_common.hpp>
#include <xpu_kernels/rms_norm/tiling.hpp>

namespace xpu_rms_prefetch {
using namespace AscendC;
using namespace RmsNorm;
using xpu_rms::RowRetainTilingData;

// Fixed candidate contract: D=20480, BF16 X/gamma/Y, FP32 rstd.
//
// UB accounting at D=20480:
//   3 * 40 KiB X row slots + 40 KiB gamma + 2 * 20 KiB Y segments
//   + (work 1 KiB + level1/2/3 3 KiB + temp/current/result 0.75 KiB)
// Total UB allocation is 204.75 KiB. The
// queue owns each slot until the row's two Y segments finish MTE3; FreeTensor
// is the reuse fence, not the vector computation.
// xFp32Buf is intentionally absent.  Reduction consumes the retained BF16
// row through ComputeFormerImplV2, and Y consumes that same row after rstd is
// available.  The X queue is not released until both Y segments have
// completed their MTE3 copy; the queue then controls slot reuse.
template <typename DX, typename DG, bool IS_GEMMA>
class KernelRmsNormPrefetch {
public:
    __aicore__ inline KernelRmsNormPrefetch() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                const RowRetainTilingData* tiling)
    {
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        blockFactor = tiling->block_factor;
        rowFactor = tiling->row_factor;
        ubFactor = tiling->ub_factor;
        ubLoop = tiling->ub_loop;
        epsilon = tiling->epsilon;
        avgFactor = tiling->avg_factor;

        curBlockIdx = GetBlockIdx();
        coreNum = GetBlockNum();
        rowWork = (curBlockIdx < coreNum - 1) ? blockFactor
                                             : numRow - (coreNum - 1) * blockFactor;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DX*>(x) + curBlockIdx * blockFactor * numCol,
                            rowWork * numCol);
        gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ DG*>(gamma), numCol);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DX*>(y) + curBlockIdx * blockFactor * numCol,
                            rowWork * numCol);
        rstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rstd) + curBlockIdx * blockFactor,
                                rowWork);

        // Static TPipe allocation.  Host launch passes num_ubuf_bytes=0.
        pipe.InitBuffer(inQueueX, 3, numColAlign * sizeof(DX));
        pipe.InitBuffer(inQueueGamma, 1, numColAlign * sizeof(DG));
        pipe.InitBuffer(outQueueY, 2, Y_SEGMENT * sizeof(DX));
        pipe.InitBuffer(outQueueRstd, 1, ROW_CAPACITY * sizeof(float));
        pipe.InitBuffer(workLocalBuf, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level1Buf, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level2Buf, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level3Buf, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(tempBuf, RmsNorm::V_LENGTH * sizeof(float));
        pipe.InitBuffer(currentWorkBuf, RmsNorm::V_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<DG> gammaLocal = inQueueGamma.DeQue<DG>();
        LocalTensor<float> resultRows = outQueueRstd.AllocTensor<float>();
        Duplicate(resultRows, 0.0f, ROW_CAPACITY);

        uint32_t nextRow = 0;
        CopyInX(nextRow++);  // prime row 0
        if (nextRow < rowWork) {
            CopyInX(nextRow++);  // prime row 1 before taking row 0
        }
        uint32_t rstdCount = 0;
        while (rstdCount < rowWork) {
            LocalTensor<DX> xLocal = inQueueX.DeQue<DX>();
            // Enqueue row+2 before vector work on the current row. With three
            // X slots, row+1 and row+2 are resident while row is computed;
            // the current slot is released only after Y.
            if (nextRow < rowWork) {
                CopyInX(nextRow++);
            }

            LocalTensor<float> currentWork = currentWorkBuf.Get<float>();
            ComputeFormerForRow(currentWork, xLocal);
            ComputeRstdAndPack(currentWork, resultRows, rstdCount);
            // ComputeYForRow reads currentWork[0] after the paired vector
            // stores in ComputeRstdAndPack have completed.
            PipeBarrier<PIPE_V>();

            ComputeYForRow(xLocal, gammaLocal, currentWork, rstdCount);
            inQueueX.FreeTensor(xLocal);
            ++rstdCount;
        }

        outQueueRstd.EnQue<float>(resultRows);
        LocalTensor<float> packedRstd = outQueueRstd.DeQue<float>();
        DataCopyImpl<float>(rstdGm, packedRstd, 1, rstdCount, 0, 0);
        outQueueRstd.FreeTensor(packedRstd);
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    static constexpr uint32_t D_FIXED = 20480;
    static constexpr uint32_t UB_FACTOR_FIXED = 8192;
    static constexpr uint32_t UB_LOOP_FIXED = 2;
    static constexpr uint32_t Y_SEGMENT = 10240;
    static constexpr uint32_t ROW_CAPACITY = 64;

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
        DataCopyImpl<DG>(gammaLocal, gammaGm, 1, static_cast<uint32_t>(numCol), 0, 0);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInX(uint32_t row)
    {
        LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
        DataCopyPadExtParams<DX> padParams{false, 0, 0, static_cast<DX>(0.0)};
        DataCopyImpl<DX>(xLocal, xGm[static_cast<uint64_t>(row) * numCol], 1,
                         static_cast<uint32_t>(numCol), 0, 0, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeFormerHandle(LocalTensor<float>& dstLocal, LocalTensor<DX>& xLocal,
                                               LocalTensor<float>& tempLocal, uint32_t srcOffset,
                                               uint32_t dstOffset, uint32_t count, uint32_t power)
    {
        uint32_t calNum = CeilAlign(static_cast<uint64_t>(count) * sizeof(DX), ALIGN_512_FACTOR) / sizeof(DX);
        LocalTensor<DX> xView = xLocal[srcOffset];
        ComputeFormerImplV2(dstLocal, xView, tempLocal, dstOffset, calNum, power);
    }

    // Same Sqrt + reciprocal Div arithmetic and avg/epsilon order as the
    // public ComputeRstd in rms_norm/arch35/rms_norm_regbase_common.h. The
    // final vector register is stored to both the Y-consumed currentWork and
    // the row-packed result buffer, avoiding scalar GetValue/SetValue and
    // V_S/S_V event transitions.
    __aicore__ inline void ComputeRstdAndPack(LocalTensor<float>& rstdLocal,
                                               LocalTensor<float>& resultRows, uint32_t row)
    {
        __local_mem__ float* rstdLocalAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* resultRowsAddr = (__ubuf__ float*)resultRows.GetPhyAddr();
        uint32_t sreg = 1;
        __VEC_SCOPE__
        {
            RegTensor<float> srcReg, dstReg, sqrtReg;
            MaskReg maskReg = UpdateMask<float>(sreg);
            MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
            DataCopy(srcReg, rstdLocalAddr);
            Muls(srcReg, srcReg, avgFactor, maskReg);
            Adds(dstReg, srcReg, epsilon, maskReg);
            Sqrt(sqrtReg, dstReg, maskReg);
            Duplicate(srcReg, float(1.0), maskReg);
            Div(dstReg, srcReg, sqrtReg, maskReg);
            DataCopy(rstdLocalAddr, dstReg, maskReg);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(resultRowsAddr + row, dstReg, maskOne);
        }
    }

    __aicore__ inline void ComputeFormerForRow(LocalTensor<float>& currentWork, LocalTensor<DX>& xLocal)
    {
        LocalTensor<float> workLocal = workLocalBuf.Get<float>();
        LocalTensor<float> level1Local = level1Buf.Get<float>();
        LocalTensor<float> level2Local = level2Buf.Get<float>();
        LocalTensor<float> level3Local = level3Buf.Get<float>();
        LocalTensor<float> tempLocal = tempBuf.Get<float>();

        Duplicate(level1Local, 0.0f, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level2Local, 0.0f, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level3Local, 0.0f, RmsNorm::ONCE_VECTOR_SIZE);
        uint32_t level1 = 0;
        uint32_t level2 = 0;
        uint32_t level3 = 0;

        // The split-D order is kept verbatim for D=20480:
        //   [0,8192)   count=8192 power=4096
        //   [8192,12288) count=4096 power=4096
        //   [12288,20480) count=8192 power=8192
        ComputeFormerHandle(tempLocal, xLocal, workLocal, 0, 0, 8192, 4096);
        ComputeFormerHandle(tempLocal, xLocal, workLocal, 8192, 1, 4096, 4096);
        ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
        ++level1;
        ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);

        ComputeFormerHandle(level1Local, xLocal, workLocal, 12288, level1, 8192, 8192);
        ++level1;
        ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        ComputeMultiLevelRstd<false>(currentWork, 0, level1Local, level2Local, level3Local, level1, level2);
    }

    __aicore__ inline void ComputeYForRow(LocalTensor<DX>& xLocal, LocalTensor<DG>& gammaLocal,
                                          LocalTensor<float>& rstdLocal, uint32_t row)
    {
        constexpr uint32_t segmentCount = (D_FIXED + Y_SEGMENT - 1) / Y_SEGMENT;
        for (uint32_t segment = 0; segment < segmentCount; ++segment) {
            uint32_t offset = segment * Y_SEGMENT;
            uint32_t count = (offset + Y_SEGMENT <= D_FIXED) ? Y_SEGMENT : D_FIXED - offset;
            LocalTensor<DX> xView = xLocal[offset];
            LocalTensor<DG> gammaView = gammaLocal[offset];
            LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();
            uint32_t calCount = CeilAlign(static_cast<uint64_t>(count) * sizeof(DX), ALIGN_512_FACTOR) /
                                sizeof(DX);
            ComputeLatterY<DX, DG>(xView, gammaView, yLocal, rstdLocal, 0, calCount,
                                   static_cast<uint8_t>(IS_GEMMA));
            outQueueY.EnQue<DX>(yLocal);
            yLocal = outQueueY.DeQue<DX>();
            // FreeTensor follows the MTE3 copy and is the slot-reuse fence.
            DataCopyImpl<DX>(yGm[static_cast<uint64_t>(row) * numCol + offset], yLocal, 1, count, 0, 0);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 3> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TBuf<TPosition::VECCALC> workLocalBuf;
    TBuf<TPosition::VECCALC> level1Buf;
    TBuf<TPosition::VECCALC> level2Buf;
    TBuf<TPosition::VECCALC> level3Buf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> currentWorkBuf;
    GlobalTensor<DX> xGm;
    GlobalTensor<DG> gammaGm;
    GlobalTensor<DX> yGm;
    GlobalTensor<float> rstdGm;
    uint64_t curBlockIdx{0};
    uint64_t coreNum{0};
    uint64_t numRow{0};
    uint64_t numCol{0};
    uint64_t numColAlign{0};
    uint64_t blockFactor{0};
    uint32_t rowFactor{0};
    uint32_t ubFactor{0};
    uint32_t ubLoop{0};
    uint32_t rowWork{0};
    float epsilon{0.0f};
    float avgFactor{0.0f};
};

}  // namespace xpu_rms_prefetch

#endif  // XPU_RMS_NORM_PREFETCH_ROW_RETAIN_KERNEL_H_
