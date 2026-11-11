"""Small GPU and NPU kernels with an explicit device implementation."""

from .rms_norm import rms_norm
from .mhc_post_backward import mhc_post_backward

__all__ = ["rms_norm", "mhc_post_backward"]
