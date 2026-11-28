"""Lazy Ascend backend for fused_quant_payload."""
def fused_quant_payload_npu(*args):
    from ._ascend import _load_extension
    return _load_extension().fused_quant_payload(*args)
