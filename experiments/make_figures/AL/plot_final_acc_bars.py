"""Final AL test accuracy, one bar per model family (mean and std over seeds).

Thin main over ``common.accbars``, everything AL-specific lives in
``experiments/make_figures/AL/runs.py``. Each run is evaluated twice, with its delays as trained
and with them rounded to integers, so the discretization cost is explicit.

Outputs:  figures/AL/final_acc_bars/
Usage:    python experiments/make_figures/AL/plot_final_acc_bars.py [--seeds 0 1 2] [--recompute]
"""

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import matplotlib.pyplot as plt

from common.accbars import run_accbars_analysis
from common.utils import save_fig
from common import style
import runs as al


def plot_discretization_cost(df, out_dir):
    """Accuracy with the delays as trained vs rounded to integers, side by side."""
    df = df.copy()
    df["acc_rounded"] = [al.read_test_accuracy(rd, "acc_rounded") for rd in df["run_dir"]]
    if df["acc_rounded"].isna().any():
        print("[warn] some runs have no cached acc_rounded, skipping the "
              "discretization figure (re-run with --recompute)")
        return

    agg = (df.groupby("model")[["test_acc", "acc_rounded"]]
             .agg(["mean", "std"]).fillna(0.0))
    order = [m for m in al.MODEL_PREFIXES if m in agg.index]
    agg = agg.loc[order]
    agg["delta"] = agg[("acc_rounded", "mean")] - agg[("test_acc", "mean")]

    spec = al.SPEC
    fig, ax = plt.subplots(figsize=style.FIGSIZE_LINE)
    xs, w = np.arange(len(agg)), 0.38
    for off, col, alpha, lab in ((-w / 2, "test_acc", 0.85, "delays as trained"),
                                 (+w / 2, "acc_rounded", 0.35, "delays rounded")):
        bars = ax.bar(xs + off, agg[(col, "mean")], w, yerr=agg[(col, "std")],
                      capsize=4, label=lab, alpha=alpha,
                      color=[spec.color(m) for m in agg.index], edgecolor="black",
                      linewidth=0.6)
        for b, mkey in zip(bars, agg.index):
            h = spec.hatch(mkey)
            if h:
                b.set_hatch(h)
    for x, d in zip(xs, agg["delta"]):
        ax.text(x, ax.get_ylim()[0], f"Δ{d:+.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels([spec.label(m) for m in agg.index])
    ax.set_ylabel(f"{al.DATASET} test accuracy (%)")
    ax.set_title(f"{al.DATASET}: cost of rounding the trained delays to integers")
    ax.set_ylim(style.ACC_YLIM)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "discretization_cost.svg"))
    agg.to_csv(os.path.join(out_dir, "discretization_cost.csv"))
    print("\n" + agg.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to evaluate (default best.pth, as perf_AL does).")
    ap.add_argument("--recompute", action="store_true",
                    help="Re-run the test-set evaluation even if final_test.json exists.")
    ap.add_argument("--forward-version", default=al.FORWARD_VERSION,
                    help=f"Recurrent kernel for the eval (default {al.FORWARD_VERSION!r}; "
                         f"synaptic families map it to 'eventdriven').")
    ap.add_argument("--out-dir", default=os.path.join(al.OUT_ROOT, "final_acc_bars"))
    args = ap.parse_args()

    print(f"AL final test accuracy → {args.out_dir}\n"
          f"  seeds={args.seeds or al.SEEDS}  ckpt={args.ckpt}  "
          f"kernel={args.forward_version}")
    df = run_accbars_analysis(al, args.out_dir, exp_dir=args.exp_dir, seeds=args.seeds,
                              ckpt=args.ckpt, recompute=args.recompute,
                              forward_version=args.forward_version)
    if not df.empty:
        plot_discretization_cost(df, args.out_dir)


if __name__ == "__main__":
    main()
