"""Delays trained and tested on the integer grid, against fractional ones.

Compares the runs that round the delays at every epoch with the otherwise
identical sweep that leaves them fractional. Integer delays are what neuromorphic
hardware implements, so rounding is the default everywhere except AL.

Outputs:  figures/HAR/std/round_compare/
Usage:    python experiments/make_figures/HAR/analyze_rounding.py
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
from common import paths
from common import style
import runs as har

# The two learned families (the only ones with a meaningful unrounded counterpart.
# Fixed families are integer-by-construction).
FAMILIES = ["ax_learned", "syn_learned"]

# Rounding regime -> free visual channel (color/marker still encode delay type).
ROUND_LS = {"rounded": "-", "unrounded": "--"}
ROUND_LABEL = {"rounded": "rounded", "unrounded": "unrounded"}


def _rounded_root(mkey):
    """The rounded (paper) sweep root for a family, from config.MODEL_INFO."""
    return har.MODEL_INFO[mkey][0]


def _resolve(root, mkey, std, exp_dir, seeds):
    """{seed: abs_run_dir} for a (family, std) under an arbitrary sweep ``root``.
    Mirrors config.resolve_run_dirs but parameterized by the root, so the same
    layout ``<root>/<cls>/std<X>/seed<N>`` serves both rounded and unrounded runs."""
    cls = har.MODEL_PREFIXES[mkey]
    base = root if os.path.isabs(root) else os.path.join(exp_dir, root)
    out = {}
    for seed in (seeds if seeds is not None else har.SEEDS):
        path = os.path.join(base, cls, f"std{std}", f"seed{seed}")
        if os.path.isdir(path):
            out[seed] = path
    return out


def collect(unrounded_root, exp_dir, seeds):
    """Per-seed final test accuracies as rows {model, rounding, std, seed, test_acc}."""
    rows = []
    for mkey in FAMILIES:
        for rounding, root in (("rounded", _rounded_root(mkey)),
                               ("unrounded", unrounded_root)):
            for std in har.STDS:
                run_dirs = _resolve(root, mkey, std, exp_dir, seeds)
                if not run_dirs:
                    print(f"[warn] no {rounding} runs for {mkey} std{std} under {root}")
                for seed, rd in sorted(run_dirs.items()):
                    acc = har.read_test_accuracy(rd)
                    if acc is None:
                        print(f"[warn] no final_test.json in {rd}")
                        continue
                    rows.append({"model": mkey, "rounding": rounding, "std": std,
                                 "seed": seed, "test_acc": acc})
    return pd.DataFrame(rows)


def plot_round_compare(df, out_dir):
    """Accuracy vs delay_std_init, one line (mean, shaded ±SEM) per (family, regime).
    delay_std_init sits on an ordinal x-axis (evenly spaced, labeled by value)."""
    if df.empty:
        print("[warn] no runs found, nothing to plot")
        return
    ticks = sorted(df["std"].unique())
    pos = {v: i for i, v in enumerate(ticks)}

    fig, ax = plt.subplots(figsize=style.FIGSIZE_LINE)
    summary = []
    for mkey in FAMILIES:
        color = style.TYPE_COLORS[style.type_of(mkey)]
        marker = style.TYPE_MARKERS[style.type_of(mkey)]
        for rounding in ("rounded", "unrounded"):
            sub = df[(df["model"] == mkey) & (df["rounding"] == rounding)]
            if sub.empty:
                continue
            agg = (sub.groupby("std")["test_acc"]
                      .agg(mean="mean", sd="std", n="count")
                      .reset_index().sort_values("std"))
            agg["sem"] = agg["sd"].fillna(0.0) / np.sqrt(agg["n"])
            agg.insert(0, "rounding", rounding)
            agg.insert(0, "model", mkey)
            summary.append(agg)

            x = [pos[v] for v in agg["std"]]
            is_unrounded = rounding == "unrounded"
            ax.plot(x, agg["mean"], color=color, linestyle=ROUND_LS[rounding],
                    marker=marker,
                    markerfacecolor=style.SURFACE if is_unrounded else color,
                    label=f"{har.SPEC.label(mkey)} ({ROUND_LABEL[rounding]})")
            ax.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                            color=color, alpha=0.18, linewidth=0)
            print(f"  {mkey:12s} {rounding:9s} std={list(agg['std'])}  "
                  f"n_seeds={list(agg['n'])}")

    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel("σ init")
    ax.set_ylabel("final test accuracy (%)")
    ax.set_title("Rounded vs unrounded delays: accuracy vs delay_std_init\n"
                 "(mean, shaded ±SEM over seeds)")
    ax.set_ylim(style.ACC_YLIM)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    cu.save_fig(fig, os.path.join(out_dir, "round_compare.svg"))
    pd.concat(summary).to_csv(os.path.join(out_dir, "round_compare.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--unrounded-root", default=paths.runs("HAR/std_init_sweep_noround"),
                    help="Sweep root of the no-rounding runs "
                         f"(default {paths.runs('HAR/std_init_sweep_noround')}).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/std/round_compare")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(har.OUT_ROOT, "std", "round_compare")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rounded-vs-unrounded comparison → {out_dir}\n"
          f"  std={har.STDS}  seeds={args.seeds or har.SEEDS}  "
          f"unrounded_root={args.unrounded_root}")

    df = collect(args.unrounded_root, args.exp_dir, args.seeds)
    plot_round_compare(df, out_dir)


if __name__ == "__main__":
    main()
