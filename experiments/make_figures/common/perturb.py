"""Shared core for input-perturbation robustness sweeps (jitter, deletion, …).

A perturbation sweep rebuilds each trained model in the σ=0 / rounded-delay
test regime, applies an input perturbation of increasing magnitude to the test
set, and records accuracy as a function of magnitude (mean ± SEM over model seeds
× perturbation seeds). The only thing that varies between analyses is the
perturbation operator ``perturb_fn(x, gen)``. Everything below, the evaluation
loop, the aggregation, and the publication SVG plots, is shared.

``common.jitter`` and ``common.deletion`` are thin facades that supply their
operator and their magnitude axis label.
"""

import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style   # importing applies the DelRec theme (rcParams)


# --------------------------------------------------------------------------- #
# Perturbed test-set evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_perturbed(loader, model, device, calc_metric, perturb_fn, seed):
    """Test-set accuracy (%) with ``perturb_fn`` applied to every batch input."""
    from delrec.utils import reset_states

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    model.eval()
    if hasattr(loader, "reset"):
        loader.reset()

    correct, total = 0, 0
    for inputs, targets in loader:
        inputs = inputs.permute(1, 0, 2).float().to(device)   # (T, B, N)
        targets = targets.to(device)
        inputs = perturb_fn(inputs, gen)
        reset_states(model=model)
        outputs = model(inputs)
        correct += calc_metric(outputs, targets)
        total += targets.size(0)
    return 100.0 * correct / total


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(rows, group_cols):
    """Mean / SEM of ``acc`` over model seeds × perturbation seeds."""
    import pandas as pd
    df = pd.DataFrame(rows)
    # Named aggregation so the acc-std column never collides with a grouping
    # column literally called "std" (HAR's delay_std_init).
    out = (df.groupby(list(group_cols))["acc"]
           .agg(mean="mean", sd="std", n="count").reset_index())
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out.drop(columns=["sd"])


# --------------------------------------------------------------------------- #
# Publication-polished plots (SVG, editable text). All styling, colors, fonts,
# spines, colormaps, comes from common.style (the repo's central graphic chart).
# --------------------------------------------------------------------------- #
_PUB_RC = style.RC


def plot_perf_vs_magnitude(summary, out_path, title, hue, order, labels, colors,
                           xlabel, ylabel="Test accuracy (%)", linestyles=None):
    """Line plot: accuracy vs perturbation magnitude, mean ± SEM band, one line per ``hue``
    value.
    """
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for hv in order:
            sub = summary[summary[hue] == hv].sort_values("magnitude")
            if sub.empty:
                continue
            c = colors.get(hv)
            ls = (linestyles or {}).get(hv, "-")
            ax.plot(sub["magnitude"], sub["mean"], marker="o", ms=4.5, lw=1.9,
                    ls=ls, color=c, label=labels.get(hv, str(hv)))
            ax.fill_between(sub["magnitude"], sub["mean"] - sub["sem"],
                            sub["mean"] + sub["sem"], color=c, alpha=0.18, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, handlelength=1.6)
        save_fig(fig, out_path)


def plot_degradation_vs_magnitude(summary, out_path, title, hue, order, labels,
                                  colors, xlabel, baseline_mag=None, linestyles=None):
    """Line plot: relative degradation ``100·(base − mean)/base`` vs perturbation magnitude,
    one line per ``hue`` value.
    """
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for hv in order:
            sub = summary[summary[hue] == hv].sort_values("magnitude")
            if sub.empty:
                continue
            bmag = sub["magnitude"].min() if baseline_mag is None else baseline_mag
            base_rows = sub[sub["magnitude"] == bmag]
            if base_rows.empty:
                continue
            base = base_rows["mean"].iloc[0]
            if base == 0:
                continue
            degr = 100.0 * (base - sub["mean"]) / base
            band = 100.0 * sub["sem"] / base
            c = colors.get(hv)
            ls = (linestyles or {}).get(hv, "-")
            ax.plot(sub["magnitude"], degr, marker="o", ms=4.5, lw=1.9,
                    ls=ls, color=c, label=labels.get(hv, str(hv)))
            ax.fill_between(sub["magnitude"], degr - band, degr + band,
                            color=c, alpha=0.18, lw=0)
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative degradation (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, handlelength=1.6)
        save_fig(fig, out_path)


def plot_magnitude_heatmap(summary, out_path, title, x_col, x_order, x_label,
                           y_label, value="mean", cbar_label="Test accuracy (%)"):
    """Heatmap of accuracy: rows = perturbation magnitude (y, increasing upward),
    columns = ``x_col`` (x), color = ``value``. Cells are annotated with the
    value. ``summary`` has columns [x_col, 'magnitude', value]."""
    mags = sorted(summary["magnitude"].unique())
    xs = list(x_order)
    M = np.full((len(mags), len(xs)), np.nan)
    for _, r in summary.iterrows():
        i = mags.index(r["magnitude"])
        if r[x_col] in xs:
            M[i, xs.index(r[x_col])] = r[value]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(0.85 * len(xs) + 2.2, 0.6 * len(mags) + 2.0))
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=style.SEQUENTIAL)
        ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_yticks(range(len(mags)), [f"{m:g}" for m in mags])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        finite = M[np.isfinite(M)]
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
        rng = (vmax - vmin) or 1.0
        for i in range(len(mags)):
            for j in range(len(xs)):
                if not np.isfinite(M[i, j]):
                    continue
                txt_c = "white" if (M[i, j] - vmin) / rng < 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color=txt_c)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label)
        save_fig(fig, out_path)


# --------------------------------------------------------------------------- #
# Fixed-vs-learned comparison plots
# --------------------------------------------------------------------------- #
def plot_perf_vs_magnitude_pair(fixed, learned, out_path, title, color,
                                xlabel, ylabel="Test accuracy (%)",
                                fixed_label="Fixed", learned_label="Learned"):
    """Accuracy vs magnitude for one (fixed, learned) pair on a single axis:
    learned solid, fixed dashed, each with a mean ± SEM band. ``fixed`` /
    ``learned`` have columns ['magnitude', 'mean', 'sem']."""
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for sub, ls, lab in ((learned, "-", learned_label), (fixed, "--", fixed_label)):
            sub = sub.sort_values("magnitude")
            if sub.empty:
                continue
            ax.plot(sub["magnitude"], sub["mean"], marker="o", ms=4.5, lw=1.9,
                    ls=ls, color=color, label=lab)
            ax.fill_between(sub["magnitude"], sub["mean"] - sub["sem"],
                            sub["mean"] + sub["sem"], color=color, alpha=0.18, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, handlelength=1.8)
        save_fig(fig, out_path)


def plot_magnitude_heatmap_compare(fixed, learned, out_path, title, x_col, x_order,
                                   x_label, y_label, cbar_label="Test accuracy (%)",
                                   panel_titles=("Fixed", "Learned", "Learned − Fixed")):
    """Three-panel heatmap comparing a fixed/learned pair: ``fixed | learned |
    Δ(learned−fixed)``.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    mags = sorted(set(fixed["magnitude"]).union(learned["magnitude"]))
    xs = list(x_order)

    def _mat(df, col):
        M = np.full((len(mags), len(xs)), np.nan)
        for _, r in df.iterrows():
            if r[x_col] in xs:
                M[mags.index(r["magnitude"]), xs.index(r[x_col])] = r[col]
        return M

    Fm, Fs = _mat(fixed, "mean"), _mat(fixed, "sem")
    Lm, Ls = _mat(learned, "mean"), _mat(learned, "sem")
    Dm = Lm - Fm
    Ds = np.sqrt(np.nan_to_num(Fs) ** 2 + np.nan_to_num(Ls) ** 2)
    Ds[~np.isfinite(Dm)] = np.nan

    acc = np.concatenate([Fm[np.isfinite(Fm)], Lm[np.isfinite(Lm)]])
    vmin, vmax = (acc.min(), acc.max()) if acc.size else (0.0, 1.0)
    acc_norm = Normalize(vmin=vmin, vmax=vmax)
    # Δ scale spans this plot's own min→max but with 0 pinned to the colormap
    # center (TwoSlopeNorm), so the neutral color always means "no difference".
    dmin = np.nanmin(Dm) if np.isfinite(Dm).any() else 0.0
    dmax = np.nanmax(Dm) if np.isfinite(Dm).any() else 1.0
    d_norm = TwoSlopeNorm(vcenter=0.0, vmin=min(dmin, -1e-9), vmax=max(dmax, 1e-9))

    panels = ((Fm, Fs, acc_norm, style.SEQUENTIAL),
              (Lm, Ls, acc_norm, style.SEQUENTIAL),
              (Dm, Ds, d_norm, style.DIVERGING))

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(
            1, 3, figsize=(3 * (0.82 * len(xs) + 0.7) + 1.4, 0.62 * len(mags) + 2.2),
            sharey=True)
        ims = []
        for k, (ax, (M, S, norm, cmap)) in enumerate(zip(axes, panels)):
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, norm=norm)
            ims.append(im)
            cmo = cmap  # a Colormap object (from common.style)
            ax.set_xticks(range(len(xs)), [str(x) for x in xs])
            ax.set_xlabel(x_label)
            ax.set_title(panel_titles[k])
            if k == 0:
                ax.set_yticks(range(len(mags)), [f"{m:g}" for m in mags])
                ax.set_ylabel(y_label)
            for i in range(len(mags)):
                for j in range(len(xs)):
                    if not np.isfinite(M[i, j]):
                        continue
                    r, g, b, _ = cmo(norm(M[i, j]))
                    txt_c = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
                    s = S[i, j] if np.isfinite(S[i, j]) else 0.0
                    ax.text(j, i, f"{M[i, j]:+.1f}\n±{s:.1f}" if k == 2
                            else f"{M[i, j]:.1f}\n±{s:.1f}",
                            ha="center", va="center", fontsize=6.5, color=txt_c)
        fig.colorbar(ims[1], ax=axes[1], fraction=0.046, pad=0.03).set_label(cbar_label)
        fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.03).set_label(
            "Δ accuracy (pts)")
        fig.suptitle(title, y=1.02)
        save_fig(fig, out_path)


def _interp_rows(M, xp, grid):
    """Interpolate every magnitude row of ``M``, sampled at the x coordinates ``xp`` onto the
    common ``grid``.
    """
    order = np.argsort(xp)
    xs_sorted = np.asarray(xp, float)[order]
    out = np.full((M.shape[0], len(grid)), np.nan)
    for i, row in enumerate(M[:, order]):
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[i] = np.interp(grid, xs_sorted[ok], row[ok])
    return out


def _isoacc_surfaces(fixed, learned, x_col, x_order, x_coords, n_interp):
    """The three surfaces one iso-accuracy panel is drawn from: the fixed and learned accuracy
    grids, their Δ, and the (X, Y) mesh they live on, over the ordinal columns ``x_order``
    (``x_coords`` None) or over the common measured-x grid both families are interpolated
    onto (see ``plot_magnitude_isoacc_overlay``).
    """
    mags = sorted(set(fixed["magnitude"]).union(learned["magnitude"]))
    xs = list(x_order)

    def _mat(df):
        M = np.full((len(mags), len(xs)), np.nan)
        for _, r in df.iterrows():
            if r[x_col] in xs:
                M[mags.index(r["magnitude"]), xs.index(r[x_col])] = r["mean"]
        return M

    Fm, Lm = _mat(fixed), _mat(learned)
    if x_coords is None:
        X, Y = np.meshgrid(range(len(xs)), range(len(mags)))
    else:
        xf, xl = (np.asarray(c, float) for c in x_coords)
        grid = np.linspace(max(xf.min(), xl.min()), min(xf.max(), xl.max()), n_interp)
        Fm, Lm = _interp_rows(Fm, xf, grid), _interp_rows(Lm, xl, grid)
        X, Y = np.meshgrid(grid, range(len(mags)))
    return X, Y, Fm, Lm, Lm - Fm, mags


def _isoacc_levels(Fm, Lm, levels):
    """The iso-accuracy contour levels: ``levels`` if given, else ~6 round values
    spanning the accuracies both families reach."""
    if levels is not None:
        return levels
    from matplotlib.ticker import MaxNLocator
    both = np.concatenate([Fm[np.isfinite(Fm)], Lm[np.isfinite(Lm)]])
    lv = MaxNLocator(nbins=6).tick_values(both.min(), both.max())
    return [v for v in lv if both.min() < v < both.max()]


def _delta_norm(*Ds):
    """A diverging norm over the Δ surfaces ``Ds``, with 0 pinned to the colormap
    center (TwoSlopeNorm) so the neutral color always means "no difference". Passing
    several surfaces yields the one norm that covers them all, how a multi-panel
    figure puts every row on a single Δ scale."""
    from matplotlib.colors import TwoSlopeNorm
    finite = [D[np.isfinite(D)] for D in Ds]
    finite = np.concatenate([f for f in finite if f.size]) if any(f.size for f in finite) \
        else np.array([0.0, 1.0])
    vmin, vmax = min(finite.min(), -1e-9), max(finite.max(), 1e-9)
    return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax), vmin, vmax


def _draw_isoacc_panel(ax, surfaces, d_norm, vmin, vmax, levels, x_coords, x_order,
                       mags, x_label, y_label, title, fixed_label, learned_label,
                       legend=True):
    """Draw one iso-accuracy overlay (Δ background + learned/fixed contours) onto
    ``ax`` using a caller-supplied Δ norm, so several panels can share one scale.
    Returns the filled-contour handle, for the caller to hang a colorbar on."""
    from matplotlib.lines import Line2D

    X, Y, Fm, Lm, Dm = surfaces
    cf = ax.contourf(X, Y, Dm, levels=np.linspace(vmin, vmax, 13),
                     cmap=style.DIVERGING, norm=d_norm, alpha=0.8)
    csL = ax.contour(X, Y, Lm, levels=levels, colors="k", linewidths=1.6,
                     linestyles="solid")
    ax.contour(X, Y, Fm, levels=levels, colors="k", linewidths=1.3,
               linestyles="dashed")
    ax.clabel(csL, inline=True, fontsize=10, fmt="%.0f")
    # Ordinal x: one tick per swept value. Rescaled x (x_coords): a real numeric
    # axis, so leave the ticks to matplotlib's locator.
    if x_coords is None:
        ax.set_xticks(range(len(x_order)), [str(x) for x in x_order])
    else:
        # Rug of the x positions actually measured, the surfaces are linearly
        # interpolated between these (``_interp_rows``), so without them the
        # field's smoothness reads as resolution it does not have. Marker fill
        # follows the contour grammar (learned solid/filled, fixed dashed/open),
        # so the existing legend covers both.
        xl_c, xf_c = x_coords[1], x_coords[0]
        trans = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
        # Held to the contours' own span: a family's runs outside the shared grid
        # (the other family's span is what cut it) would otherwise autoscale x and
        # open a margin of empty panel next to marks with no surface under them.
        xlim = ax.get_xlim()
        ax.plot(xl_c, [0.015] * len(xl_c), color="k", marker="^", ms=5, ls="none",
                transform=trans, zorder=5)
        ax.plot(xf_c, [0.015] * len(xf_c), mfc="none", mec="k", mew=1.2, marker="^",
                ms=5, ls="none", transform=trans, zorder=5)
        ax.set_xlim(xlim)
    ax.set_yticks(range(len(mags)), [f"{m:g}" for m in mags])
    ax.set_xlabel(x_label)
    # Both wrapped to the panel: at this size the longer magnitude labels are
    # taller than the axes, and a title that overhangs the axes runs into the
    # (rotated) y-label above the panel. Short labels are returned unchanged.
    ax.set_ylabel("\n".join(textwrap.wrap(y_label, 30)))
    ax.set_title("\n".join(textwrap.wrap(title, 34)))
    # Backed rather than frameless: the contours run right under this corner, and
    # the labels are unreadable straight on top of them.
    if legend:
        # On a measured-x axis the handles carry the rug marker too, so one entry per
        # family explains both its contours and its sample positions.
        mk = dict(marker="^", ms=5) if x_coords is not None else {}
        ax.legend(handles=[Line2D([0], [0], color="k", ls="-", lw=1.6,
                                  label=learned_label, **mk),
                           Line2D([0], [0], color="k", ls="--", lw=1.3,
                                  label=fixed_label, mfc="none", **mk)],
                  loc="upper right", frameon=True, framealpha=0.85,
                  facecolor=style.SURFACE, edgecolor="none")
    return cf


def plot_magnitude_isoacc_overlay(fixed, learned, out_path, title, x_col, x_order,
                                  x_label, y_label, levels=None,
                                  cbar_label="Δ accuracy (pts)",
                                  fixed_label="Fixed", learned_label="Learned",
                                  x_coords=None, n_interp=60):
    """Overlay of iso-accuracy contours for a fixed/learned pair on a Δ background. Filled
    background is Δ(learned − fixed) (diverging, centered at 0).
    """
    X, Y, Fm, Lm, Dm, mags = _isoacc_surfaces(fixed, learned, x_col, x_order,
                                              x_coords, n_interp)
    levels = _isoacc_levels(Fm, Lm, levels)
    d_norm, vmin, vmax = _delta_norm(Dm)

    # Sized as a paper panel: RC's theme at RC_LARGE_TEXT's sizes on a smaller canvas,
    # so the text still reads once the panel is placed in a figure at ~1:1.
    with plt.rc_context({**_PUB_RC, **style.RC_LARGE_TEXT}):
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        cf = _draw_isoacc_panel(ax, (X, Y, Fm, Lm, Dm), d_norm, vmin, vmax, levels,
                                x_coords, x_order, mags, x_label, y_label, title,
                                fixed_label, learned_label)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).set_label(cbar_label)
        save_fig(fig, out_path)


def plot_magnitude_isoacc_overlay_rows(rows, out_path, title, x_col, x_order,
                                       y_label, levels=None,
                                       cbar_label="Δ accuracy (pts)",
                                       fixed_label="Fixed", learned_label="Learned",
                                       n_interp=60):
    """One figure stacking several ``plot_magnitude_isoacc_overlay`` panels as rows, e.g.
    axonal on top, synaptic below, so the two delay types are read as one panel.
    """
    surf = [_isoacc_surfaces(f, l, x_col, x_order, xc, n_interp)
            for _, f, l, _, xc in rows]
    levels = _isoacc_levels(np.concatenate([s[2].ravel() for s in surf]),
                            np.concatenate([s[3].ravel() for s in surf]), levels)
    d_norm, vmin, vmax = _delta_norm(*[s[4] for s in surf])

    # Constrained layout: each row carries its own x label and title (the measured-x
    # axes differ per row, so they cannot be shared away), and at this text size they
    # collide with the neighboring panel under the default spacing.
    with plt.rc_context({**_PUB_RC, **style.RC_LARGE_TEXT}):
        fig, axes = plt.subplots(len(rows), 1, figsize=(4.8, 3.8 * len(rows)),
                                 layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, (row_title, _, _, x_label, x_coords), s in zip(axes, rows, surf):
            X, Y, Fm, Lm, Dm, mags = s
            cf = _draw_isoacc_panel(ax, (X, Y, Fm, Lm, Dm), d_norm, vmin, vmax, levels,
                                    x_coords, x_order, mags, x_label, y_label,
                                    row_title, fixed_label, learned_label,
                                    legend=(ax is axes[0]))
        # One colorbar for the whole stack. The scale is shared, so a per-row bar
        # would repeat the same axis len(rows) times.
        fig.colorbar(cf, ax=list(axes), fraction=0.046, pad=0.03).set_label(cbar_label)
        fig.suptitle("\n".join(textwrap.wrap(title, 44)))
        save_fig(fig, out_path)
