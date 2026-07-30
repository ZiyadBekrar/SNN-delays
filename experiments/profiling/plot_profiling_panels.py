"""Paper panels for the two kernel-profiling runs.

Re-plots ``profile_kernels.py`` (isolated-layer sweeps) and
``profile_kernels_datasets.py`` (real batches), keeping the three curves per delay type
that carry the message: the PyTorch reference ``v1``, the fused kernel
(``triton_exact`` axonal, ``eventdriven`` synaptic), and the delay-free
``vanilla_recurrent`` floor. Both figures are forward+backward, the training regime.

Outputs:  kernel_sweeps.svg, kernel_datasets.svg              [measurements]
Usage:    MPLBACKEND=Agg python experiments/profiling/plot_profiling_panels.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Central graphic chart (experiments/make_figures/common/style.py applies its rcParams on import).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

from common import paths          # noqa: E402  (needs the path above)
from common import style

# --------------------------------------------------------------------------- #
# Inputs / outputs
# --------------------------------------------------------------------------- #
def _latest(prefix):
    """Newest profiling run of a kind, or None. These measurements are
    hardware-specific, so pick up whichever run is present rather than naming one
    machine's timestamped directory."""
    import glob
    runs = sorted(glob.glob(paths.runs("profiling", f"{prefix}_*")))
    return Path(runs[-1]) if runs else None


SWEEP_RUN = _latest("sweeps")
DATASET_RUN = _latest("datasets")
OUT_DIR = Path(paths.figures("profiling"))

REGIME = "fwdbwd"          # both figures show the full training step
DATASETS = ["SSC", "HAR", "PSMNIST"]

# Operating point held fixed while one axis is swept (profile_kernels.DEFAULTS),
# marked on every panel so the columns can be read against a common reference.
DEFAULTS = dict(T=250, B=256, N=256, D=25, sigma=1.0)
SWEEPS = {
    "T":     [50, 100, 250, 500, 1000],
    "B":     [32, 64, 128, 256, 512],
    "N":     [64, 128, 256, 512, 1024],
    "D":     [5, 10, 25, 50, 100],
    "sigma": [0.0, 1.0, 2.0, 4.0, 8.0],
}
# Plain text, no mathtext: matplotlib renders mathtext glyph-by-glyph in a
# different font stack, so a "$T$" here would not match the y-axis labels.
XLABEL = {
    "T": "Sequence length  T",
    "B": "Batch size  B",
    "N": "Layer width  N",
    "D": "Max recurrent delay  D",
    "sigma": "Delay spread  σ",
}
LOGX = {"T": True, "B": True, "N": True, "D": True, "sigma": False}

# --------------------------------------------------------------------------- #
# The three curves per delay type.
# Color = delay type (the repo chart: axonal purple, synaptic red). The fast
# kernel is the saturated hue + solid, the PyTorch reference a light tint +
# dashed, and the delay-free floor gray/dotted (the repo's control style).
# (key, legend label, color, marker, linestyle)
# --------------------------------------------------------------------------- #
def _tint(hex_c):
    return style._mix(hex_c, 0.48)


MODES = {
    "axonal": [
        ("v1", "PyTorch delays (v1)", _tint(style.BLUE), "s", "--"),
        ("triton_exact", "Triton exact scan", style.BLUE, "o", "-"),
        ("vanilla", "no-delay RSNN (PyTorch)", style.INK_MUTED, "P", ":"),
    ],
    "synaptic": [
        ("v1", "PyTorch delays (v1)", _tint(style.ORANGE), "s", "--"),
        ("eventdriven", "Triton event-driven", style.ORANGE, "D", "-"),
        ("vanilla", "no-delay RSNN (PyTorch)", style.INK_MUTED, "P", ":"),
    ],
}
KIND_TITLE = {
    "axonal": "Axonal delays  (one delay per neuron)",
    "synaptic": "Synaptic delays  (one delay per synapse)",
}
METRICS = [("time_ms", "per-batch time (ms)"), ("mem_mib", "peak memory (MiB)")]


# --------------------------------------------------------------------------- #
# Panel set 1, isolated-layer sweeps
# --------------------------------------------------------------------------- #
def _sweep_panel(ax, df, var, metric, modes):
    """One (axis x metric) panel: three curves, log-log, OOM drop-outs marked."""
    sub = df[(df["sweep"] == var) & (df["regime"] == REGIME)]
    for key, label, color, marker, ls in modes:
        d = sub[sub["mode"] == key].sort_values("value")
        ax.plot(d["value"], d[metric], marker=marker, ls=ls, color=color,
                label=label, lw=1.6, ms=4.6, markeredgecolor="white",
                markeredgewidth=0.5, clip_on=False, zorder=3)
        # Points re-measured by profile_kernels_bigN.py under a narrower W tile:
        # same kernel, non-default launch config -> hollow marker.
        fill = d[d["from_fill"]]
        if len(fill):
            ax.plot(fill["value"], fill[metric], marker=marker, ls="none", ms=5,
                    mfc="white", mec=color, mew=1.4, clip_on=False, zorder=5)
        # Kernels that ran out of GPU memory / Triton resources are NaN in the
        # CSV: show where the curve dies rather than letting it end silently.
        miss = d[d[metric].isna()]["value"]
        if len(miss):
            ax.plot([miss.iloc[0]], [ax.get_ylim()[1]], marker="x", ms=5.5,
                    color=color, mew=1.5, clip_on=False, zorder=4)

    ax.axvline(DEFAULTS[var], color=style.BASELINE, lw=1.0, ls="-", zorder=0)
    ax.set_yscale("log")
    if LOGX[var]:
        ax.set_xscale("log")
    # Only first / middle / last are labeled. The panels are ~1.4 in wide on a
    # portrait page, so all five swept values would collide. Every value still
    # carries a marker.
    vals = SWEEPS[var]
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" if i in (0, len(vals) // 2, len(vals) - 1) else ""
                        for i, v in enumerate(vals)])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.5)
    ax.margins(x=0.08)


# Portrait page: rows = swept axis, columns = {time, memory} within each of the
# two delay-type groups. Slightly smaller absolute point sizes than RC, on an
# 8.3 in wide figure (vs 14 in) they still read considerably larger on the page.
RC_SWEEPS = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
GROUP_X = {"axonal": (0.105, 0.515), "synaptic": (0.585, 0.995)}
GRID_TOP, GRID_BOTTOM = 0.885, 0.088


def plot_sweeps(dfs, out_dir):
    with plt.rc_context(RC_SWEEPS):
        fig = plt.figure(figsize=(8.3, 11.0))
        axes = {}
        for kind, (left, right) in GROUP_X.items():
            gs = fig.add_gridspec(5, 2, left=left, right=right, top=GRID_TOP,
                                  bottom=GRID_BOTTOM, hspace=0.42, wspace=0.34)
            axes[kind] = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                                   for r in range(5)])

        for kind, ax_grid in axes.items():
            for r, var in enumerate(SWEEPS):
                for c, (metric, ylabel) in enumerate(METRICS):
                    ax = ax_grid[r, c]
                    _sweep_panel(ax, dfs[kind], var, metric, MODES[kind])
                    if r == 0:
                        ax.set_title(ylabel, fontsize=9.5,
                                     color=style.INK_SECONDARY, pad=6)

        # One y-scale per metric across BOTH delay types, so a height means the
        # same thing everywhere in the figure.
        for c, (metric, _) in enumerate(METRICS):
            col = [ax for g in axes.values() for ax in g[:, c]]
            lo = min(ax.get_ylim()[0] for ax in col)
            hi = max(ax.get_ylim()[1] for ax in col)
            for ax in col:
                ax.set_ylim(lo, hi)

        # Row labels (the swept axis) in the left margin, once per row.
        for r, var in enumerate(SWEEPS):
            pos = axes["axonal"][r, 0].get_position()
            fig.text(0.022, (pos.y0 + pos.y1) / 2, XLABEL[var], rotation=90,
                     ha="center", va="center", fontsize=10.5,
                     color=style.INK_SECONDARY)

        for kind, (left, right) in GROUP_X.items():
            fig.text((left + right) / 2, 0.968, KIND_TITLE[kind], fontsize=11,
                     fontweight="bold", color=style.INK_PRIMARY,
                     ha="center", va="center")
            handles = [Line2D([], [], color=c, marker=m, ls=ls, lw=1.6, ms=4.6,
                              markeredgecolor="white", markeredgewidth=0.5, label=lab)
                       for _, lab, c, m, ls in MODES[kind]]
            fig.legend(handles=handles, loc="center", ncol=1, frameon=False,
                       bbox_to_anchor=((left + right) / 2, 0.928),
                       handlelength=2.4, labelspacing=0.35)

        held = ", ".join(f"{k if k != 'sigma' else 'σ'}={v:g}"
                         for k, v in DEFAULTS.items())
        fig.text(0.5, 0.052,
                 f"Single recurrent layer, forward+backward, A40. One axis swept at "
                 f"a time around {held}\n(gray line); median of 10 runs. Markers sit "
                 f"on all five swept values; only three are labeled.\n"
                 f"Hollow marker: backward re-run with a narrower W tile (BK=32) — "
                 f"the default BK=64 exceeds\nthe A40's 99 KiB of shared memory.   "
                 f"×: no tile fits at all.",
                 ha="center", va="top", linespacing=1.6, fontsize=8,
                 color=style.INK_MUTED)

        for ext in ("svg", "png"):
            fig.savefig(out_dir / f"kernel_sweeps.{ext}")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Panel set 2, real datasets
# --------------------------------------------------------------------------- #
def _bar_panel(ax, df, kind, metric, ylabel):
    modes = MODES[kind]
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.26
    for i, (key, label, color, _, _) in enumerate(modes):
        vals = [float(df[(df["dataset"] == ds) & (df["kind"] == kind) &
                         (df["regime"] == REGIME) & (df["mode"] == key)][metric].iloc[0])
                for ds in DATASETS]
        pos = x + (i - 1) * width
        ax.bar(pos, vals, width * 0.92, color=color, label=label, zorder=3,
               edgecolor="white", linewidth=0.6,
               hatch=("///" if key == "v1" else None))
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:,.0f}", (p, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5,
                        color=style.INK_SECONDARY)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="major", alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.28)


def plot_datasets(df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    for r, kind in enumerate(("axonal", "synaptic")):
        for c, (metric, ylabel) in enumerate(METRICS):
            _bar_panel(axes[r, c], df, kind, metric, ylabel)
        axes[r, 0].set_title(KIND_TITLE[kind], fontsize=11, fontweight="bold",
                             loc="left", pad=26)
        handles = [Patch(facecolor=c, edgecolor="white",
                         hatch=("///" if k == "v1" else None), label=lab)
                   for k, lab, c, _, _ in MODES[kind]]
        axes[r, 0].legend(handles=handles, ncol=3, frameon=False, fontsize=9,
                          loc="lower left", bbox_to_anchor=(0.0, 1.005),
                          columnspacing=1.2, handlelength=1.4)

    fig.text(0.5, 0.015,
             "Full model on real batches, forward+backward, A40, at each dataset's "
             "own operating point (T, B, widths from its config), median of 3 runs "
             "× 4 batches.",
             ha="center", fontsize=8.5, color=style.INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"kernel_datasets.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def load_sweep(kind):
    """The sweep CSV, with any `kernel_profile_bigN.csv` fill-ins substituted in."""
    df = pd.read_csv(SWEEP_RUN / kind / "kernel_profile.csv")
    df["from_fill"] = False
    path = SWEEP_RUN / kind / "kernel_profile_bigN.csv"
    if path.exists():
        fill = pd.read_csv(path)
        fill["from_fill"] = True
        key = ["sweep", "value", "regime", "mode"]
        df = df[~df.set_index(key).index.isin(fill.set_index(key).index)]
        df = pd.concat([df, fill], ignore_index=True)
    return df


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {k: load_sweep(k) for k in ("axonal", "synaptic")}
    plot_sweeps(dfs, out_dir)

    df_ds = pd.read_csv(DATASET_RUN / "bench_datasets.csv")
    plot_datasets(df_ds, out_dir)
    print(f"Wrote kernel_sweeps.svg/.png and kernel_datasets.svg/.png to {out_dir}")


if __name__ == "__main__":
    main()
