"""Dispatcher routing attention through Triton JVP kernels or torch SDPA.

The public entry point :func:`triton_attn_func` accepts ``q, k, v`` of shape
``(B, L, H, D)`` and returns an output of the same shape. When invoked under
forward-mode AD (dual tensors / ``torch.func.jvp``), the primal output and its
tangent are computed together by the hand-written Triton kernels in
:mod:`.triton_kernels`. Otherwise it falls back to
:func:`torch.nn.functional.scaled_dot_product_attention` (SDPA), which already
supports standard reverse-mode autograd.

Notes
-----
- Only non-causal attention is supported.
- The dual path requires ``seq_len_q`` divisible by ``block_q`` (128).
- Reverse-mode gradients flow through the SDPA primal; the Triton kernels are
  only used to supply the forward-mode tangent needed by the ESD/LSD losses.
"""

from __future__ import annotations

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .triton_kernels import (
    _attn_fwd_dual,
    _attn_fwd_dual_split,
    split_head_dim,
)

__all__ = ['triton_attn_func']

_BLOCK_Q = 128
_BLOCK_KV = 128
_NUM_STAGES = 2
_NUM_WARPS = 8


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Run reference SDPA attention for ``(B, L, H, D)`` inputs.

    Transposes to ``(B, H, L, D)``, runs non-causal scaled dot-product attention,
    and transposes back to ``(B, L, H, D)``.
    """
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qt, kt, vt, scale=softmax_scale)
    return out.transpose(1, 2)


def _launch_dual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Triton forward+JVP kernel for ``(B, L, H, D)`` inputs.

    Returns
    -------
    out : torch.Tensor
        Primal attention output, shape ``(B, L, H, D)``.
    t_out : torch.Tensor
        Forward-mode tangent of the output, same shape.
    """
    b, l_q, h, d = q.shape
    l_kv = k.shape[1]
    grid = (l_q // _BLOCK_Q, b * h, 1)

    out = torch.empty_like(q)
    t_out = torch.empty_like(tq)
    m = torch.empty((b, h, l_q), dtype=torch.float32, device=q.device)

    tile_dim, tail_dim = split_head_dim(d)
    has_tail = tail_dim > 0

    strides = (
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *tq.stride(),
        *tk.stride(),
        *tv.stride(),
        *out.stride(),
        *t_out.stride(),
    )

    if has_tail:
        _attn_fwd_dual_split[grid](
            q,
            k,
            v,
            tq,
            tk,
            tv,
            m,
            out,
            t_out,
            softmax_scale,
            *strides,
            h,
            l_q,
            l_kv,
            d,
            tile_dim,
            tail_dim,
            block_q=_BLOCK_Q,
            block_kv=_BLOCK_KV,
            num_stages=_NUM_STAGES,
            num_warps=_NUM_WARPS,
        )
    else:
        _attn_fwd_dual[grid](
            q,
            k,
            v,
            tq,
            tk,
            tv,
            m,
            out,
            t_out,
            softmax_scale,
            *strides,
            h,
            l_q,
            l_kv,
            d,
            block_q=_BLOCK_Q,
            block_kv=_BLOCK_KV,
            num_stages=_NUM_STAGES,
            num_warps=_NUM_WARPS,
        )
    return out, t_out


def _is_dual(*tensors: torch.Tensor) -> bool:
    """Return ``True`` if any tensor carries a forward-mode tangent."""
    for tensor in tensors:
        _, tangent = fwAD.unpack_dual(tensor)
        if tangent is not None:
            return True
    return False


class _DMFAttentionJVP(torch.autograd.Function):
    """Put DMF's Triton JVP behind a storage-backed autograd boundary."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        tv: torch.Tensor,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, tangent = _launch_dual(q, k, v, tq, tk, tv, softmax_scale)
        return out, tangent

    @staticmethod
    def setup_context(ctx: object, inputs: tuple, output: tuple) -> None:
        q, k, v, *_, softmax_scale = inputs
        ctx.save_for_backward(q, k, v)
        ctx.softmax_scale = softmax_scale

    @staticmethod
    def backward(
        ctx: object, grad_out: torch.Tensor | None, grad_tangent: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        del grad_tangent
        if grad_out is None:
            return (None,) * 7
        q, k, v = ctx.saved_tensors
        q_grad = q.detach().requires_grad_(True)
        k_grad = k.detach().requires_grad_(True)
        v_grad = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _sdpa(q_grad, k_grad, v_grad, ctx.softmax_scale)
        grads = torch.autograd.grad(out, (q_grad, k_grad, v_grad), grad_out)
        return *grads, None, None, None, None


def triton_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Attention with Triton-computed forward-mode JVP, SDPA fallback otherwise.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key, value tensors of shape ``(B, L, H, D)``.
    softmax_scale : float, optional
        Softmax scale. Defaults to ``head_dim ** -0.5``.

    Returns
    -------
    torch.Tensor
        Attention output of shape ``(B, L, H, D)``. If the inputs are dual
        tensors, the result is a dual tensor whose tangent was computed by the
        Triton JVP kernel.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if not _is_dual(q, k, v):
        return _sdpa(q, k, v, softmax_scale)

    q_p, q_t = fwAD.unpack_dual(q)
    k_p, k_t = fwAD.unpack_dual(k)
    v_p, v_t = fwAD.unpack_dual(v)
    q_t = torch.zeros_like(q_p) if q_t is None else q_t
    k_t = torch.zeros_like(k_p) if k_t is None else k_t
    v_t = torch.zeros_like(v_p) if v_t is None else v_t

    out, t_out = _DMFAttentionJVP.apply(
        q_p, k_p, v_p, q_t, k_t, v_t, softmax_scale
    )
    return fwAD.make_dual(out, t_out)
