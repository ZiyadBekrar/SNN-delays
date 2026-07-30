"""Two alternative delay discretizations, against nearest-integer rounding.

The learnable delays live on a continuous parameter but are used on an integer
grid, so how they are discretized is a free choice. Compares the deterministic
nearest rounding used throughout with a straight-through estimator (round in the
forward, gradient to the fractional copy) and stochastic rounding (round up with
probability equal to the fractional part, so a sub-step drift is not erased by
the deadzone of nearest rounding).

Outputs:  figures/HAR/std/rounding_alternatives/
Usage:    python experiments/make_figures/HAR/analyze_rounding_alternatives.py
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
from matplotlib.lines import Line2D

import common.utils as cu
from common import paths
from common import style
import runs as har

# Root of the alternatives sweep (the 'nearest' baseline comes from config.MODEL_INFO).
ROUND_SWEEP_ROOT = paths.runs("HAR/rounding_alternatives_sweep")

# The baseline mode: it has no <mode>_std<X> subtree, it is the std sweep.
BASELINE_MODE = "nearest"

# Panels: (delay type, model key). Only the learned families discretize anything,
# the fixed families are integer by construction.
PANELS = [("axonal", "ax_learned"), ("synaptic", "syn_learned")]

# Rounding mode -> (legend label, color, line style). Color encodes the mode here
# (the panel already fixes the delay type). 'stochastic' takes the chart's 3rd
# categorical hue, as the common family does in analyze_commondelay.py.
MODE_STYLE = {
    BASELINE_MODE: ("nearest (baseline)", style.BLUE, "-"),
    "ste":         ("STE",                style.ORANGE, "--"),
    "stochastic":  ("stochastic",         "#1baf7a", "-."),
}
MODES = list(MODE_STYLE)

# Two panels side by side, sized like analyze_commondelay's single paper panel.
FIGSIZE_PANELS = (9.6, 4.4)


def resolve_run_dirs(mkey, mode, std, exp_dir=".", seeds=None, root=ROUND_SWEEP_ROOT):
    """{seed: run_dir} for a (learned family, rounding mode, std). ``nearest`` falls
    through to the std sweep (config.resolve_run_dirs). The alternatives live at
    ``<root>/<cls>/<mode>_std<X>/seed<N>``, the ``mode_tag`` of the sweep script."""
    if mode == BASELINE_MODE:
        return har.resolve_run_dirs(mkey, std, exp_dir, seeds)
    cls = har.MODEL_PREFIXES[mkey]
    base = root if os.path.isabs(root) else os.path.join(exp_dir, root)
    out = {}
    for seed in (seeds if seeds is not None else har.SEEDS):
        path = os.path.join(base, cls, f"{mode}_std{std}", f"seed{seed}")
        if os.path.isdir(path):
            out[seed] = path
    return out


def collect(modes, stds, exp_dir, seeds, root):
    """Per-seed final test accuracies as rows {delay_type, model, mode, std, seed, acc}."""
    rows = []
    for dtype, mkey in PANELS:
        for mode in modes:
            for std in stds:
                run_dirs = resolve_run_dirs(mkey, mode, std, exp_dir, seeds, root)
                if not run_dirs:
                    print(f"[warn] no {mode} runs for {mkey} std{std}")
                for seed, rd in sorted(run_dirs.items()):
                    acc = har.read_test_accuracy(rd)
                    if acc is None:
                        print(f"[warn] no final_test.json in {rd}")
                        continue
                    rows.append({"delay_type": dtype, "model": mkey, "mode": mode,
                                 "std": std, "seed": seed, "test_acc": acc})
    return pd.DataFrame(rows)


def plot_rounding_alternatives(df, modes, out_dir):
    """One panel per delay type: accuracy vs delay_std_init, one line (mean, shaded
    ±SEM over seeds) per rounding mode. The panels share the accuracy axis, so the
    axonal and synaptic gaps between modes are read at the same scale. The swept
    values sit on an ordinal x (evenly spaced, labeled by value)."""
    if df.empty:
        print("[warn] no runs found, nothing to plot")
        return
    ticks = sorted(df["std"].unique())
    pos = {v: i for i, v in enumerate(ticks)}
    summary = []

    with plt.rc_context(style.RC_LARGE_TEXT):
        fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE_PANELS, sharey=True)
        for ax, (dtype, mkey) in zip(np.atleast_1d(axes), PANELS):
            marker = style.TYPE_MARKERS[dtype]
            for mode in modes:
                label, color, ls = MODE_STYLE[mode]
                sub = df[(df["model"] == mkey) & (df["mode"] == mode)]
                if sub.empty:
                    continue
                agg = (sub.groupby("std")["test_acc"]
                          .agg(mean="mean", sd="std", n="count")
                          .reset_index().sort_values("std"))
                agg["sem"] = agg["sd"].fillna(0.0) / np.sqrt(agg["n"])
                agg.insert(0, "mode", mode)
                agg.insert(0, "delay_type", dtype)
                summary.append(agg)

                x = [pos[v] for v in agg["std"]]
                ax.plot(x, agg["mean"], color=color, linestyle=ls, marker=marker,
                        label=label)
                ax.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                                color=color, alpha=0.18, linewidth=0)

            ax.set_xticks(list(pos.values()))
            ax.set_xticklabels([format(v, har.STD_AXIS.value_fmt) for v in ticks])
            ax.set_xlabel(har.STD_AXIS.label)
            ax.set_title(dtype)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("final test accuracy (%)")

        # One key for both panels (the mode set is identical in each), above the
        # figure so it never covers the curves. The keys are drawn marker-free on
        # purpose: the marker encodes the delay type, so copying a panel's handle
        # would put an axonal ○ next to a label that holds in both panels.
        handles = [Line2D([], [], color=MODE_STYLE[m][1], linestyle=MODE_STYLE[m][2])
                   for m in modes]
        fig.legend(handles, [MODE_STYLE[m][0] for m in modes], loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure,
                   ncol=len(modes), frameon=False)
        fig.tight_layout()
        # Inside the rc_context: savefig re-measures the text (bbox="tight").
        cu.save_fig(fig, os.path.join(out_dir, "rounding_alternatives.svg"))

    summary = pd.concat(summary, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "rounding_alternatives.csv"), index=False)
    for dtype, mkey in PANELS:
        for mode in modes:
            sub = summary[(summary["delay_type"] == dtype) & (summary["mode"] == mode)]
            if sub.empty:
                continue
            accs = " ".join(f"{v:.2f}" for v in sub["mean"])
            print(f"  {dtype:9s} {mode:11s} std={list(sub['std'])} "
                  f"n_seeds={list(sub['n'])}  acc={accs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--root", default=ROUND_SWEEP_ROOT,
                    help=f"Sweep root of the alternative-rounding runs "
                         f"(default {ROUND_SWEEP_ROOT}).")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=MODES,
                    help=f"Rounding modes to plot. Default: {MODES}.")
    ap.add_argument("--values", type=int, nargs="+", default=None,
                    help="Restrict to these delay_std_init values. Default: config.STDS.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/std/rounding_alternatives")
    args = ap.parse_args()

    stds = args.values or har.STDS
    out_dir = args.out_dir or os.path.join(har.OUT_ROOT, "std", "rounding_alternatives")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rounding-alternatives comparison → {out_dir}\n"
          f"  modes={args.modes}  std={stds}  seeds={args.seeds or har.SEEDS}  "
          f"root={args.root}")

    df = collect(args.modes, stds, args.exp_dir, args.seeds, args.root)
    plot_rounding_alternatives(df, args.modes, out_dir)


if __name__ == "__main__":
    main()
