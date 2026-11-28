"""Lazy Ascend backend for fused_send_slots."""
def fused_send_slots_npu(*args):
    from ._ascend import _load_extension
    return _load_extension().fused_send_slots(*args)
