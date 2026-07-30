"""Input-gradient reach: how far back in time a gradient survives.

Computes the norm of the gradient reaching each past input step. Two probes are
recorded. The cross-entropy probe is the one the paper defines. Its output-error
vector scales by orders of magnitude as a model trains, so only scale-invariant
readings of it compare across models. The unit-vector Jacobian probe
backpropagates a fixed vector instead, making the magnitudes directly comparable.

Models are put in train() mode but with dropout and batch-norm disabled: the fused
kernels take an inference path under eval() that detaches the input graph, which
would return no input gradient at all.
"""

import numpy as np
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style   # importing applies the DelRec theme (rcParams)


# --------------------------------------------------------------------------- #
# Probe loss + per-timestep input gradient
# --------------------------------------------------------------------------- #
def final_step_ce(out, targets):
    """Cross-entropy on the last-timestep readout ``out[-1]``, the reach probe loss.
    ``out`` is the model's ``(T, B, n_classes)`` logits. Unlike the time-averaged training
    loss, credit reaches earlier inputs only through the recurrence, so ``g(t)`` is a
    genuine credit-assignment-reach profile."""
    import torch.nn.functional as F
    return F.cross_entropy(out[-1], targets)


def unit_vector_probe(n_classes, n_probes=4, seed=0, device=None):
    """Fixed random unit vectors in readout space, the output side of the Jacobian probe."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    U = torch.randn(n_probes, n_classes, generator=gen)
    U = U / U.norm(dim=1, keepdim=True)
    return U.to(device) if device is not None else U


def jacobian_grad_over_time(model, x, U):
    """Scale-free reach probe: ``g(t) = mean_i mean_b ||Jₜᵀuᵢ||`` for fixed unit ``uᵢ``. For
    the final-step probe ``x_t`` reaches the loss only through ``out[T-1]``, so ``dL/dx_t =
    Jₜᵀ v`` with ``Jₜ = ∂out[T-1]/∂x_t`` and ``v = dL/dout[T-1]``.
    """
    curves = [input_grad_over_time(model, x, None,
                                   calc_loss=lambda out, _t, u=u: (out[-1] * u).sum())
              for u in U]
    return np.mean(np.stack(curves, axis=0), axis=0)


def set_probe_mode(model):
    """Put ``model`` in the gradient-probe regime and return it."""
    import torch
    from spikingjelly.activation_based import layer as sj_layer
    model.train()
    for m in model.modules():
        if isinstance(m, (sj_layer.Dropout, torch.nn.Dropout, torch.nn.BatchNorm1d)):
            m.eval()
    return model


def input_grad_per_sample(model, x, targets, calc_loss=final_step_ce):
    """Per-sample, per-timestep input-gradient norm ``||dL/dx_{t,b}||`` for one ``(T, B, C)``
    batch.
    """
    from delrec.utils import reset_states

    x = x.detach().requires_grad_(True)
    reset_states(model=model)
    out = model(x)
    loss = calc_loss(out, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return x.grad.detach().norm(dim=2).cpu().numpy()   # (T,B,C) -> (T,B)


def input_grad_over_time(model, x, targets, calc_loss=final_step_ce):
    """Batch-mean per-timestep input-gradient norm ``g(t) = mean_b ||dL/dx_{t,b}||``.
    Length-``T`` numpy array. Convenience wrapper over :func:`input_grad_per_sample`."""
    return input_grad_per_sample(model, x, targets, calc_loss).mean(axis=1)


# --------------------------------------------------------------------------- #
# Gradient-quality scalars derived from g(t)
# --------------------------------------------------------------------------- #
def effective_reach(g):
    """Credit-assignment length from ``g(t)``."""
    g = np.asarray(g, dtype=np.float64)
    T = g.shape[0]
    k = (T - 1) - np.arange(T)                       # time-before-output per t
    total = g.sum()
    centroid = float((k * g).sum() / total) if total > 0 else np.nan

    # tau: order g by ascending k (i.e. Reverse time) and fit log g on positive entries.
    gk = g[::-1]                                      # gk[k] = g at time-before-output k
    kk = np.arange(T, dtype=np.float64)
    pos = gk > 0
    if pos.sum() >= 2:
        slope = np.polyfit(kk[pos], np.log(gk[pos]), 1)[0]
        tau = float(-1.0 / slope) if slope < 0 else np.inf
    else:
        tau = np.nan
    return {"tau": tau, "centroid": centroid}


def vanishing_ratio(g, window=None):
    """Gradient at the earliest vs latest input timesteps: ``mean(g[:w]) / mean(g[-w:])``."""
    g = np.asarray(g, dtype=np.float64)
    T = g.shape[0]
    w = window if window is not None else max(1, T // 20)
    late = g[-w:].mean()
    return float(g[:w].mean() / late) if late > 0 else np.inf


# --------------------------------------------------------------------------- #
# Aggregation over batches / seeds
# --------------------------------------------------------------------------- #
def aggregate(curves):
    """Mean + SEM over a list/stack of length-``T`` curves. Returns ``(mean, sem)``
    (both length ``T``). SEM is zero for a single curve."""
    arr = np.stack([np.asarray(c, dtype=np.float64) for c in curves], axis=0)
    mean = arr.mean(axis=0)
    sem = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
           if arr.shape[0] > 1 else np.zeros_like(mean))
    return mean, sem


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_grad_over_time(curves, out_path, *, title="Input-gradient reach g(t)",
                        xlabel="Timestep t  (readout at t = T-1)",
                        ylabel="Input gradient  ||dL/dx_t||", logy=True,
                        ms_per_step=None):
    """Overlaid per-timestep gradient curves with ±SEM bands.

    ``curves``: ordered dict/list of ``(label, mean, sem, color, ls)`` tuples (``sem`` may
    be ``None``). ``ms_per_step`` adds a secondary top axis in milliseconds (HAR: 50)."""
    fig, ax = plt.subplots(figsize=style.FIGSIZE_LINE)
    for label, mean, sem, color, ls in curves:
        t = np.arange(len(mean))
        (line,) = ax.plot(t, mean, label=label, color=color, ls=ls or "-", lw=1.8)
        if sem is not None:
            ax.fill_between(t, mean - sem, mean + sem, color=line.get_color(), alpha=0.22)
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8)
    if ms_per_step is not None:
        secax = ax.secondary_xaxis("top", functions=(lambda s: s * ms_per_step,
                                                      lambda ms: ms / ms_per_step))
        secax.set_xlabel("Time (ms)")
    fig.tight_layout()
    return save_fig(fig, out_path)
