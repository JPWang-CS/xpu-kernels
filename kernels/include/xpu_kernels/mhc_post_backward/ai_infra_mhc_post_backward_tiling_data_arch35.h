/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ai_infra_mhc_post_backward_tiling_data.h
 * \brief ai_infra_mhc_post_backward
 */

#ifndef AI_INFRA_MHC_POST_BACKWARD_TILING_DATA_ARCH35_H
#define AI_INFRA_MHC_POST_BACKWARD_TILING_DATA_ARCH35_H

class AiInfraMhcPostBackwardOptimizedTilingDataArch35 {
public:
    uint32_t totalItems{0};
    uint32_t usedAic{0};
    uint32_t n{0};
    uint32_t D{0};
    uint32_t tileD{0};
    uint32_t cubeBatchGroup{0};
    uint32_t cubeL1K{0};
    uint32_t cubeBaseK{0};
    uint32_t cubeKPartitionCount{1};
    uint32_t cubeKPartitionSize{0};
};

#endif // AI_INFRA_MHC_POST_BACKWARD_TILING_DATA_ARCH35_H
