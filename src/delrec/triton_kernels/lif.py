"""LIF neuron dynamics as @triton.jit step helpers.

These are the per-neuron-type plug-in point for the library: each kernel's
inner time-step calls `lif_fwd_step` / `lif_bwd_step` instead of inlining the
charge/fire/reset arithmetic. Triton resolves and inlines these jit functions at
compile time, so the generated code (and therefore the numerics) is identical
to writing the body directly in the kernel.

A new neuron type (ALIF, adaptive threshold, Izhikevich, ...) provides its own
`*_fwd_step` / `*_bwd_step` pair with the same shape contract: elementwise over
the `(BLOCK_B, BLOCK_N)` program tile, branching on `tl.constexpr` flags only.

Only the pure neuron dynamics live here. The kernels keep everything that is
scan/recurrence plumbing: the dropout fusion `a = xv + dv * r`, the recurrence
gradient mix `gY = gy + gfut`, and the scan carry `lam_v = decay * du`.
"""

import triton
import triton.language as tl

from .surrogate import _surrogate_grad


@triton.jit
def lif_fwd_step(v, a, decay, charge_offset, v_th, v_reset_val, SOFT: tl.constexpr):
    """One forward LIF step (decay_input=False)."""
    u = decay * v + charge_offset + a                              # pre-reset membrane
    spike = (u - v_th >= 0).to(tl.float32)
    if SOFT:
        v_new = u - spike * v_th
    else:
        v_new = u - spike * (u - v_reset_val)
    return u, spike, v_new


@triton.jit
def lif_bwd_step(gY, u, lam_v, v_th, v_reset_val, gamma, alpha,
                 SOFT: tl.constexpr, DETACH: tl.constexpr, SURR: tl.constexpr):
    """One backward LIF step."""
    xg = u - v_th
    s = (xg >= 0).to(tl.float32)
    sg = _surrogate_grad(xg, gamma, alpha, SURR)

    if SOFT:
        if DETACH:
            gS = gY
        else:
            gS = gY - v_th * lam_v
        du = lam_v + gS * sg
    else:  # hard reset: v = u - s*(u - v_reset)
        if DETACH:
            gS = gY
        else:
            gS = gY - lam_v * (u - v_reset_val)
        du = lam_v * (1.0 - s) + gS * sg
    return du
