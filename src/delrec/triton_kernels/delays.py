"""Differentiable delay-kernel and tap builders (pure PyTorch, no triton.jit).

The triangular delay kernel `build_mask` is the soft relaxation h_{sigma,d} of
Eq. (7) used by the exact forward. `build_lag` / `build_lag_interp` produce the
integer / 2-tap hard delays for the rounded and STE-interp forwards. `build_syn_mask` is the per-synapse counterpart of `build_mask`. All are
differentiable w.r.t. d / p_spread so the autograd.Functions can map dmask back
to dd / dp via autograd.grad.
"""

import torch


def build_mask(d, sigma, p_spread, use_sig_p):
    """Triangular delay kernel, identical to multi_step_forward_v2.

    Returns (mask (N, L), L). Differentiable w.r.t. d / p_spread.
    """
    device, dtype = d.device, d.dtype
    if use_sig_p:
        s = 1.0 + 2.0 * sigma * torch.sigmoid(p_spread).to(device=device, dtype=dtype)
        s_max = s.max()
        s = s.unsqueeze(1)                                            # (N, 1)
    else:
        s = torch.tensor(1.0 + float(sigma), device=device, dtype=dtype)
        s_max = s
    L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
    support = torch.arange(L, device=device, dtype=dtype)            # (L,)
    mask = torch.clamp(s - torch.abs(support[None, :] - (1.0 + d)[:, None]), min=0.0) / (s * s)
    return mask, L                                                    # (N, L)


def build_lag(d, L):
    """Per-neuron integer delay tap for the approx variant: lag = round(d)+1,
    clamped to [1, L-1]."""
    lag = (torch.round(d).long() + 1).clamp_(1, max(1, L - 1))
    return lag.to(torch.int32)


def build_lag_interp(d, L):
    """Per-neuron 2-tap linear-interpolation delay for the STE-interp forward."""
    pos = 1.0 + d
    lag_lo = torch.floor(pos).long().clamp_(1, max(1, L - 1))
    frac = (pos - lag_lo.to(d.dtype)).clamp_(0.0, 1.0)
    return lag_lo.to(torch.int32), frac.to(torch.float32)



def build_syn_mask(d, sigma, p_spread, use_sig_p):
    """Per-synapse triangular delay kernel."""
    device, dtype = d.device, d.dtype
    N = d.shape[1]
    if use_sig_p:
        s = 1.0 + 2.0 * sigma * torch.sigmoid(p_spread).to(device=device, dtype=dtype)  # (N,)
        s_max = s.max()
        s = s.view(1, N, 1)                                          # (1, N_out, 1)
    else:
        s = torch.tensor(1.0 + float(sigma), device=device, dtype=dtype)
        s_max = s
    L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
    support = torch.arange(L, device=device, dtype=dtype)            # (L,)
    mask = torch.clamp(
        s - torch.abs(support[None, None, :] - (1.0 + d)[:, :, None]), min=0.0) / (s * s)
    return mask, L                                                    # (N_in, N_out, L)
