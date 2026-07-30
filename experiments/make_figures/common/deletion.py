"""Deletion: remove input samples independently at random.

Each input value is dropped with probability p, Bernoulli thinning of the spike
counts, or zeroing of a (time step, channel) reading on a continuous signal,
where on z-scored input zero is roughly the channel mean.

Magnitude is p. The identity at zero. Distinct from occlusion, which removes one
contiguous block: this probes tolerance to scattered loss.
"""

import torch

# Re-exported so the sweep scripts import everything deletion-related from here.
from common.perturb import (  # noqa: F401
    evaluate_perturbed, aggregate, plot_perf_vs_magnitude, plot_magnitude_heatmap,
    plot_perf_vs_magnitude_pair, plot_magnitude_heatmap_compare,
    plot_magnitude_isoacc_overlay,
)

# One label per input regime: on a continuous-signal dataset (HAR) the deleted unit is a
# scalar sample, not a spike, so the spike wording would be plainly false on its figures.
DELETION_XLABELS = {"spikes": "Input spike-deletion probability p",
                    "signal": "Input sample-deletion probability p"}
DELETION_XLABEL = DELETION_XLABELS["spikes"]  # default of the plot wrappers below (SSC)


# --------------------------------------------------------------------------- #
# Deletion operators
# --------------------------------------------------------------------------- #
def _delete_spikes(x, p, gen):
    """Drop each input spike independently with probability ``p`` (Bernoulli
    thinning). ``x`` : (T, B, N) non-negative spike counts. Counts > 1 are
    expanded so each unit spike is dropped on its own."""
    nz = x.nonzero(as_tuple=False)                        # (K, 3): t, b, n
    if nz.numel() == 0:
        return x
    counts = x[nz[:, 0], nz[:, 1], nz[:, 2]].round().long()  # (K,)
    rep = torch.repeat_interleave(nz, counts, dim=0)         # (S, 3), one row per spike
    keep = torch.rand(rep.shape[0], generator=gen, device=x.device) >= p
    rep = rep[keep]
    out = torch.zeros_like(x)
    if rep.numel():
        out.index_put_((rep[:, 0], rep[:, 1], rep[:, 2]),
                       torch.ones(rep.shape[0], dtype=x.dtype, device=x.device),
                       accumulate=True)
    return out


def _delete_signal(x, p, gen):
    """Zero each ``(timestep, channel)`` input value independently with probability
    ``p``. ``x`` : (T, B, C). Each scalar reading is a deletable unit (the analog
    of one spike). On z-scored input, zeroing ≈ replacing with the channel mean."""
    keep = (torch.rand(x.shape, generator=gen, device=x.device) >= p).to(x.dtype)
    return x * keep


def delete_input(x, p, mode, gen):
    """Apply spike/signal deletion of probability ``p`` to ``x`` (T,B,·).
    ``p <= 0`` is the identity (so the p=0 point is the clean input)."""
    if p is None or p <= 0:
        return x
    if mode == "spikes":
        return _delete_spikes(x, p, gen)
    if mode == "signal":
        return _delete_signal(x, p, gen)
    raise ValueError(f"Unknown deletion mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Deletion-flavoured wrappers over the shared perturbation core
# --------------------------------------------------------------------------- #
def evaluate_deleted(loader, model, device, calc_metric, p, mode, deletion_seed):
    """Test-set accuracy (%) with spike/signal deletion ``p`` applied per batch."""
    return evaluate_perturbed(loader, model, device, calc_metric,
                              lambda x, g: delete_input(x, p, mode, g), deletion_seed)


def plot_perf_vs_deletion(summary, out_path, title, hue, order, labels, colors,
                          ylabel="Test accuracy (%)"):
    """Line plot of accuracy vs deletion probability (see plot_perf_vs_magnitude)."""
    plot_perf_vs_magnitude(summary, out_path, title, hue, order, labels, colors,
                           DELETION_XLABEL, ylabel)


def plot_deletion_heatmap(summary, out_path, title, x_col, x_order,
                          x_label="delay_std_init", value="mean",
                          cbar_label="Test accuracy (%)"):
    """Heatmap of accuracy over (x_col, deletion probability) (see plot_magnitude_heatmap)."""
    plot_magnitude_heatmap(summary, out_path, title, x_col, x_order, x_label,
                           DELETION_XLABEL, value, cbar_label)


def plot_perf_vs_deletion_pair(fixed, learned, out_path, title, color,
                               ylabel="Test accuracy (%)", **kw):
    """Accuracy vs deletion for one fixed/learned pair (see plot_perf_vs_magnitude_pair)."""
    plot_perf_vs_magnitude_pair(fixed, learned, out_path, title, color,
                                DELETION_XLABEL, ylabel, **kw)


def plot_deletion_heatmap_compare(fixed, learned, out_path, title, x_col, x_order,
                                  x_label="delay_std_init", **kw):
    """Three-panel fixed|learned|Δ deletion heatmap (see plot_magnitude_heatmap_compare)."""
    plot_magnitude_heatmap_compare(fixed, learned, out_path, title, x_col, x_order,
                                   x_label, DELETION_XLABEL, **kw)


def plot_deletion_isoacc_overlay(fixed, learned, out_path, title, x_col, x_order,
                                 x_label="delay_std_init", **kw):
    """Iso-accuracy overlay (learned solid / fixed dashed) over a Δ background
    (see plot_magnitude_isoacc_overlay)."""
    plot_magnitude_isoacc_overlay(fixed, learned, out_path, title, x_col, x_order,
                                  x_label, DELETION_XLABEL, **kw)
