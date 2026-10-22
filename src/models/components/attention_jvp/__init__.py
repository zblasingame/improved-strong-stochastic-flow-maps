"""Triton fused-attention forward kernels with JVP (forward-mode AD) support.

This subpackage provides a GPU attention path whose forward-mode automatic
differentiation (``torch.func.jvp`` / dual tensors) is computed by hand-written
Triton kernels. The Eulerian and Lagrangian self-distillation losses differentiate
*through* the network with forward-mode AD; stock fused-attention kernels do not
support this, hence the custom JVP kernels.

The whole subpackage is import-gated: if Triton is unavailable (e.g. a CPU-only
box), :data:`triton_attn_func` is ``None`` and callers fall back to the
``base`` / ``torch_sdpa`` attention paths.

References
----------
- Official Triton fused attention tutorial.
- JVP support adapted from Ryu1845 and Birch-san gists (see ``triton_kernels``).
"""

try:
    from .dispatcher import triton_attn_func
except ImportError:  # triton not installed / unavailable
    triton_attn_func = None

__all__ = ['triton_attn_func']
