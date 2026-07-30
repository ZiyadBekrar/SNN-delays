"""Temporal jitter: perturb when each input arrives.

Draws a Gaussian offset per sample and time step. On a continuous signal the
signal is resampled at the jittered times, with the offset shared across
channels. On spike trains each spike is moved by a rounded offset.

Magnitude is the standard deviation in time steps. The identity at zero.
"""

import torch

# Re-exported so the sweep scripts import everything jitter-related from here.
from common.perturb import (  # noqa: F401
    evaluate_perturbed, aggregate, plot_perf_vs_magnitude, plot_magnitude_heatmap,
    plot_perf_vs_magnitude_pair, plot_magnitude_heatmap_compare,
    plot_magnitude_isoacc_overlay,
)

JITTER_XLABEL = "Input temporal jitter σ (timesteps)"


# --------------------------------------------------------------------------- #
# Jitter operators
# --------------------------------------------------------------------------- #
def _jitter_spikes(x, sigma, gen):
    """Move each input spike independently by round(N(0, σ)) timesteps."""
    T = x.shape[0]
    nz = x.nonzero(as_tuple=False)                        # (K, 3): t, b, n
    if nz.numel() == 0:
        return x
    counts = x[nz[:, 0], nz[:, 1], nz[:, 2]].round().long()  # (K,)
    rep = torch.repeat_interleave(nz, counts, dim=0)         # (S, 3), one row per spike
    off = torch.round(torch.randn(rep.shape[0], generator=gen, device=x.device) * sigma).long()
    new_t = (rep[:, 0] + off).clamp_(0, T - 1)
    out = torch.zeros_like(x)
    out.index_put_((new_t, rep[:, 1], rep[:, 2]),
                   torch.ones(rep.shape[0], dtype=x.dtype, device=x.device),
                   accumulate=True)
    return out


def _jitter_signal(x, sigma, gen):
    """Resample a continuous signal in time: out[t] = x[clamp(round(t+N(0,σ)))]."""
    T, B, C = x.shape
    off = torch.round(torch.randn(T, B, generator=gen, device=x.device) * sigma).long()
    base = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
    idx = (base + off).clamp_(0, T - 1)                    # (T, B)
    return x.gather(0, idx.unsqueeze(-1).expand(T, B, C))


def jitter_input(x, sigma, mode, gen):
    """Apply temporal jitter of magnitude ``sigma`` (timesteps) to ``x`` (T,B,·).
    ``sigma <= 0`` is the identity (so the σ=0 point is the clean input)."""
    if sigma is None or sigma <= 0:
        return x
    if mode == "spikes":
        return _jitter_spikes(x, sigma, gen)
    if mode == "signal":
        return _jitter_signal(x, sigma, gen)
    raise ValueError(f"Unknown jitter mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Jitter-flavoured wrappers over the shared perturbation core
# --------------------------------------------------------------------------- #
def evaluate_jittered(loader, model, device, calc_metric, sigma, mode, jitter_seed):
    """Test-set accuracy (%) with temporal jitter ``sigma`` applied to every batch."""
    return evaluate_perturbed(loader, model, device, calc_metric,
                              lambda x, g: jitter_input(x, sigma, mode, g), jitter_seed)


def plot_perf_vs_jitter(summary, out_path, title, hue, order, labels, colors,
                        ylabel="Test accuracy (%)"):
    """Line plot of accuracy vs jitter magnitude (see plot_perf_vs_magnitude)."""
    plot_perf_vs_magnitude(summary, out_path, title, hue, order, labels, colors,
                           JITTER_XLABEL, ylabel)


def plot_jitter_heatmap(summary, out_path, title, x_col, x_order,
                        x_label="delay_std_init", value="mean",
                        cbar_label="Test accuracy (%)"):
    """Heatmap of accuracy over (x_col, jitter magnitude) (see plot_magnitude_heatmap)."""
    plot_magnitude_heatmap(summary, out_path, title, x_col, x_order, x_label,
                           JITTER_XLABEL, value, cbar_label)


def plot_perf_vs_jitter_pair(fixed, learned, out_path, title, color,
                             ylabel="Test accuracy (%)", **kw):
    """Accuracy vs jitter for one fixed/learned pair (see plot_perf_vs_magnitude_pair)."""
    plot_perf_vs_magnitude_pair(fixed, learned, out_path, title, color,
                                JITTER_XLABEL, ylabel, **kw)


def plot_jitter_heatmap_compare(fixed, learned, out_path, title, x_col, x_order,
                                x_label="delay_std_init", **kw):
    """Three-panel fixed|learned|Δ jitter heatmap (see plot_magnitude_heatmap_compare)."""
    plot_magnitude_heatmap_compare(fixed, learned, out_path, title, x_col, x_order,
                                   x_label, JITTER_XLABEL, **kw)


def plot_jitter_isoacc_overlay(fixed, learned, out_path, title, x_col, x_order,
                               x_label="delay_std_init", **kw):
    """Iso-accuracy overlay (learned solid / fixed dashed) over a Δ background
    (see plot_magnitude_isoacc_overlay)."""
    plot_magnitude_isoacc_overlay(fixed, learned, out_path, title, x_col, x_order,
                                  x_label, JITTER_XLABEL, **kw)
