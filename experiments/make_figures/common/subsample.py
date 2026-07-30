"""Temporal subsampling: keep one input sample in every k.

Coarsens the input's temporal resolution without changing its length. On a
continuous signal the gaps hold the last kept sample. On spike trains the counts
are binned into windows of width k, so the total spike count is preserved.

Deterministic, and the identity at k = 1. Distinct from time-warping, which
changes the speed but keeps the resolution: this asks whether delays learned at
one input clock survive a coarser one.
"""

import torch

from common.perturb import evaluate_perturbed  # noqa: F401  (re-exported)

SUBSAMPLE_XLABEL = "Temporal subsampling factor k  (keep 1 / k samples)"


def _subsample_signal(x, k):
    """Zero-order-hold decimation of a continuous signal: out[t] = x[k*(t//k)]."""
    T = x.shape[0]
    t = torch.arange(T, device=x.device)
    idx = (t // k) * k                                      # (T,) bin-start indices
    return x.index_select(0, idx)


def _subsample_signal_impulse(x, k):
    """Impulse decimation: keep the samples at t = 0, k, 2k, …. Mute the gaps."""
    T = x.shape[0]
    t = torch.arange(T, device=x.device)
    keep = (t % k == 0).view(-1, *([1] * (x.dim() - 1)))    # (T,1,…) broadcast mask
    return x * keep.to(x.dtype)


def _subsample_spikes(x, k):
    """Bin spike counts into windows of width k, summed onto each window's first
    timestep (total spike count preserved, temporal precision reduced to k)."""
    T = x.shape[0]
    t = torch.arange(T, device=x.device)
    bin_start = (t // k) * k                                # (T,)
    out = torch.zeros_like(x)
    out.index_add_(0, bin_start, x)                         # out[jk] += sum over the window
    return out


def subsample_input(x, k, mode, gen):
    """Apply temporal subsampling of factor ``k`` to ``x`` (T,B,·). ``k <= 1`` is
    the identity (the k=1 point is the clean input). The op is deterministic. ``gen`` is accepted only for a uniform interface."""
    if k is None or k <= 1:
        return x
    if mode == "signal":
        return _subsample_signal(x, k)
    if mode == "signal_impulse":
        return _subsample_signal_impulse(x, k)
    if mode == "spikes":
        return _subsample_spikes(x, k)
    raise ValueError(f"Unknown subsample mode: {mode!r}")


def evaluate_subsampled(loader, model, device, calc_metric, k, mode, seed):
    """Test-set accuracy (%) with temporal subsampling of factor ``k`` applied per
    batch. The op is deterministic, but ``seed`` is accepted for a uniform interface."""
    return evaluate_perturbed(loader, model, device, calc_metric,
                              lambda x, g: subsample_input(x, k, mode, g), seed)
