"""Final test accuracy against the swept value, one line per delay family.

Reads each run's recorded final accuracy, no rebuild, no forward pass. Swept
values sit on an ordinal axis so that clustered small values stay legible.

Outputs:  figures/HAR/<sweep>/final_acc_compare/
Usage:    python experiments/make_figures/HAR/plot_final_acc_compare.py --sweep std
"""

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common.utils as cu
from common import style
import runs as har


_spread = har.delay_spread  # the sweep figures label their x axis with the same measure

# The violin figure is a paper panel: it is placed in a multi-panel figure and so gets
# scaled down to a column, which shrinks the text with it. Only the ratio of ink to canvas
# survives that scaling, so every size here is quoted relative to the canvas width via
# PANEL_SCALE rather than in absolute points: widening the canvas to fit the violins would
# otherwise silently halve the rendered text.
#
# The width is set by the violin panel rather than the accuracy one: it packs four
# families × five swept values, and below ~16in the violins render as needles whose shapes
# can't be told apart.
FIGSIZE_PANEL = (19.0, 15.2)
# Sizes below were tuned on a 10.6in-wide canvas. Scale them with the width so they keep
# the same rendered size once the panel is placed in the figure.
PANEL_SCALE = FIGSIZE_PANEL[0] / 10.6
RC_PANEL = {
    **style.RC_LARGE_TEXT,
    "font.size": 20 * PANEL_SCALE,
    "axes.labelsize": 23 * PANEL_SCALE,
    "legend.fontsize": 18 * PANEL_SCALE,
    "xtick.labelsize": 20 * PANEL_SCALE,
    "ytick.labelsize": 20 * PANEL_SCALE,
    "lines.linewidth": 2.6 * PANEL_SCALE,
    "lines.markersize": 10 * PANEL_SCALE,
    "hatch.linewidth": 1.4 * PANEL_SCALE,
}

# The line figure's canvas. Panel-sized (small enough that RC_LARGE_TEXT reads at ~1:1
# once placed in a paper figure) and portrait: five ordinal ticks need little width, and
# the height is what separates the four accuracy curves.
FIGSIZE_LINE_PANEL = (4.4, 7.2)

# Delay ticks for the √ panel. Even spacing in √ space would land on 0, 4, 16, 36, 64,
# 100. These are the round numbers nearest that, so the labels read as delays while
# their spacing still shows the scale is warped. Ticks above the data are dropped.
SQRT_TICKS = [0, 5, 10, 20, 30, 50, 75, 100]


def _kde_bw(kde):
    """Scott's factor, floored so the kernel is never narrower than one timestep."""
    std = float(np.std(kde.dataset))
    return max(kde.scotts_factor(), 1.0 / std) if std > 0 else kde.scotts_factor()


def _init_spread(run_dir):
    """Delay spread of the run's saved init draw (None if no init_delays.npz)."""
    path = os.path.join(run_dir, "init_delays.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return _spread([z[k] for k in z.files])


def collect(axis, exp_dir, seeds, ckpt):
    """Walk the (family × swept value × seed) grid once, reading each run's accuracy
    (final_test.json), final delays (``ckpt``) and init delays (init_delays.npz).
    """
    rows, runs = [], {}
    for mkey in axis.families:
        for value in axis.values:
            run_dirs = axis.resolve(mkey, value, exp_dir, seeds)
            if not run_dirs:
                print(f"[warn] no runs for {mkey} {axis.value_dir(value)} under {exp_dir}")
            for seed, rd in sorted(run_dirs.items()):
                acc = har.read_test_accuracy(rd)
                if acc is None:
                    print(f"[warn] no final_test.json in {rd}")
                    continue
                init_std = _init_spread(rd)
                if init_std is None:
                    print(f"[warn] no init_delays.npz in {rd}")
                delays = [l["delays"] for l in cu.load_recurrent_params(rd, ckpt)]
                runs.setdefault((mkey, value), {})[seed] = delays
                rows.append({"model": mkey, axis.col: value, "seed": seed, "test_acc": acc,
                             "delay_std": _spread(delays),
                             "init_std": np.nan if init_std is None else init_std})
    return pd.DataFrame(rows), runs


def _agg(sub, col, value_col):
    """Across-seed mean / std / n / SEM of ``value_col``, one row per swept value."""
    agg = (sub.groupby(col)[value_col]
              .agg(mean="mean", sd="std", n="count")
              .reset_index().sort_values(col))
    agg["sem"] = agg["sd"].fillna(0.0) / np.sqrt(agg["n"])
    return agg


def plot_final_acc(df, axis, spec, out_dir):
    """Two stacked panels sharing x: final test accuracy (top) and final delay std (bottom),
    one line (mean, shaded ±SEM) per family.
    """
    if df.empty:
        print("[warn] no runs found, nothing to plot")
        return
    ticks = sorted(df[axis.col].unique())
    pos = {v: i for i, v in enumerate(ticks)}

    with plt.rc_context(style.RC_LARGE_TEXT):
        fig, (ax, ax_d) = plt.subplots(2, 1, figsize=FIGSIZE_LINE_PANEL, sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 1]})
        summary, delay_summary = [], []
        for mkey in axis.families:
            sub = df[df["model"] == mkey]
            if sub.empty:
                continue
            color = spec.color(mkey)
            is_fixed = style.condition_of(mkey) == "fixed"
            kw = dict(color=color, linestyle=spec.linestyle(mkey),
                      marker=style.TYPE_MARKERS[style.type_of(mkey)],
                      markerfacecolor=style.SURFACE if is_fixed else color)

            for panel, value_col, store in ((ax, "test_acc", summary),
                                            (ax_d, "delay_std", delay_summary)):
                agg = _agg(sub, axis.col, value_col)
                agg.insert(0, "model", mkey)
                store.append(agg)
                x = [pos[v] for v in agg[axis.col]]
                # Only the top panel is labeled: both panels share the family styling,
                # so one legend serves the figure.
                panel.plot(x, agg["mean"], label=spec.label(mkey) if panel is ax else None,
                           **kw)
                panel.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                                   color=color, alpha=0.18, linewidth=0)
            print(f"  {mkey:12s} {axis.col}={list(summary[-1][axis.col])}  "
                  f"n_seeds={list(summary[-1]['n'])}")

        # The delay std actually realized at init, pooled over families and seeds: the
        # reference the frozen fixed families must sit on, and the line the learned
        # families move away from.
        ref = df.groupby(axis.col)["init_std"].mean().dropna()
        if not ref.empty:
            ax_d.plot([pos[v] for v in ref.index], ref.values, color=style.INK_MUTED,
                      linestyle=":", linewidth=1.2, zorder=0, label="at init")

        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([axis.pretty(v) for v in ticks])
        ax_d.set_xlabel(axis.label)
        ax.set_ylabel("final test accuracy (%)")
        ax_d.set_ylabel("final delay std (steps)")
        ax.set_ylim(style.ACC_YLIM)
        ax.grid(alpha=0.3)
        ax_d.grid(alpha=0.3)
        # Above the panel: at panel width a four-entry legend inside the axes covers the
        # lines. The delay panel's own key names only the init reference, and goes
        # top-left, the corner its rising lines leave free.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax_d.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        # Inside the rc_context: savefig re-measures the text (bbox="tight").
        cu.save_fig(fig, os.path.join(out_dir, "final_acc_compare.svg"))
    pd.concat(summary).to_csv(os.path.join(out_dir, "final_acc_compare.csv"), index=False)
    pd.concat(delay_summary).to_csv(os.path.join(out_dir, "final_delay_std.csv"), index=False)



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint the final delays are read from (default best.pth, "
                         "the one final_test.json was computed on).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/final_acc_compare")
    args = ap.parse_args()

    axis = har.get_axis(args)
    out_dir = args.out_dir or axis.out_dir("final_acc_compare")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Final-accuracy comparison ({axis.key} sweep) → {out_dir}\n"
          f"  {axis.col}={axis.values}  seeds={args.seeds or har.SEEDS}")

    df, runs = collect(axis, args.exp_dir, args.seeds, args.ckpt)
    plot_final_acc(df, axis, har.SPEC, out_dir)


if __name__ == "__main__":
    main()
