"""Lazy Ascend backend for fused_owner_combine."""
def fused_owner_combine_npu(*args):
    from ._ascend import _load_extension
    return _load_extension().fused_owner_combine(*args)
