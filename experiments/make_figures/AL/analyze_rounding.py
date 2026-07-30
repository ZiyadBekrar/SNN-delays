"""What projecting the AL delays onto the integer grid costs.

AL is the one benchmark trained without per-epoch rounding, so its learned delays
stay fractional. Three regimes are compared: kept fractional throughout, rounded
at test time only, and rounded at every epoch during training. The middle one
mixes the cost of discretization with a train/test mismatch. The third isolates
the cost of the constraint itself.

Only the learned families appear. Fixed delays are integer from their
initialization, so rounding is a no-op for them.

Outputs:  figures/AL/round_compare/
Usage:    python experiments/make_figures/AL/analyze_rounding.py
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import common.utils as cu
from common import paths
from common import style
import runs as al

# The two learned families. The only ones for which rounding is not a no-op.
FAMILIES = ["ax_learned", "syn_learned"]

# Default root of the trained-with-rounding runs (parallel_sweep_AL_rounding.py).
ROUNDING_ROOT = paths.runs("AL/rounding_sweep")

# regime -> (legend label, bar alpha, bar hatch). Color still encodes the delay type,
# so the regime rides on the two free channels: alpha = the delays used at test are not
# the ones the model was trained with, hatch = trained under the rounding constraint.
REGIMES = {
    "fractional": ("trained unrounded, tested as trained", 0.85, None),
    "posthoc":    ("trained unrounded, rounded at test",    0.35, None),
    "rounded":    ("trained rounded (each epoch)",          0.85, "///"),
}
ORDER = ["fractional", "posthoc", "rounded"]
BASE_REGIME = "fractional"  # the regime the deltas are taken against

# Bins of the frac(d) histogram over [0, 1). 20 -> 0.05 timestep resolution, which the
# axonal families (5 seeds x 304 delays) still populate at ~75 counts per bin.
FRAC_BINS = 20


def _rounding_run_dirs(mkey, rounding_root, exp_dir, seeds):
    """{seed: run_dir} for a family under the rounding sweep's flat layout
    ``<root>/<ModelClass>/seed<N>`` (contrast the published runs' timestamped dirs,
    which ``config.resolve_run_dirs`` globs)."""
    base = rounding_root if os.path.isabs(rounding_root) \
        else os.path.join(exp_dir, rounding_root)
    out = {}
    for seed in (seeds if seeds is not None else al.SEEDS):
        path = os.path.join(base, al.MODEL_PREFIXES[mkey], f"seed{seed}")
        if os.path.isdir(path):
            out[seed] = path
    return out


def collect(rounding_root, exp_dir, seeds):
    """Per-seed final test accuracies as rows {model, regime, seed, test_acc}.

    The first two regimes are two keys of one published run's final_test.json (the
    same checkpoint, evaluated twice). The third is a different run altogether."""
    rows = []
    for mkey in FAMILIES:
        sources = (
            ("fractional", al.resolve_run_dirs(mkey, exp_dir, seeds), "acc"),
            ("posthoc",    al.resolve_run_dirs(mkey, exp_dir, seeds), "acc_rounded"),
            ("rounded",    _rounding_run_dirs(mkey, rounding_root, exp_dir, seeds), "acc"),
        )
        for regime, run_dirs, key in sources:
            if not run_dirs:
                print(f"[warn] no {regime} runs for {mkey}")
            for seed, rd in sorted(run_dirs.items()):
                acc = al.read_test_accuracy(rd, key)
                if acc is None:
                    print(f"[warn] no '{key}' in {rd}/final_test.json — run "
                          f"experiments/make_figures/AL/plot_final_acc_bars.py first")
                    continue
                rows.append({"model": mkey, "regime": regime, "seed": seed,
                             "test_acc": acc})
    return pd.DataFrame(rows)


def plot_round_compare(df, out_dir):
    """One bar per (family, regime): mean, +/- std over seeds, annotated with the
    delta to the as-trained regime. Writes ``round_compare.svg`` + ``.csv``."""
    if df.empty:
        print("[warn] no runs found, nothing to plot")
        return

    agg = (df.groupby(["model", "regime"])["test_acc"]
             .agg(mean="mean", sd="std", n="count").reset_index())
    agg["sd"] = agg["sd"].fillna(0.0)          # a single seed has no spread
    agg["sem"] = agg["sd"] / np.sqrt(agg["n"])
    base = {m: g.set_index("regime")["mean"].get(BASE_REGIME)
            for m, g in agg.groupby("model")}
    agg["delta"] = [r["mean"] - base[r["model"]] for _, r in agg.iterrows()]

    cell = {(r["model"], r["regime"]): r for _, r in agg.iterrows()}
    order = [m for m in FAMILIES if m in set(agg["model"])]
    regimes = [g for g in ORDER if any((m, g) in cell for m in order)]

    # Thinner hatch strokes than the 1.0 pt default: at a ~3 in panel width the
    # repo's "///" otherwise reads as a black bar rather than a texture.
    with plt.rc_context({**style.RC_PAPER, "hatch.linewidth": 0.55}):
        fig, ax = plt.subplots(figsize=(3.3, 2.9), constrained_layout=True)
        xs = np.arange(len(order))
        w = 0.8 / len(regimes)
        top = 0.0
        for gi, regime in enumerate(regimes):
            _label, alpha, hatch = REGIMES[regime]
            off = (gi - (len(regimes) - 1) / 2) * w
            present = [m for m in order if (m, regime) in cell]
            means = [cell[(m, regime)]["mean"] for m in present]
            sds = [cell[(m, regime)]["sd"] for m in present]
            pos = [xs[order.index(m)] + off for m in present]
            ax.bar(pos, means, w, yerr=sds, capsize=2, alpha=alpha,
                   color=[al.SPEC.color(m) for m in present], hatch=hatch,
                   edgecolor="black", linewidth=0.6, error_kw=dict(elinewidth=0.7))
            for x, m, s, mkey in zip(pos, means, sds, present):
                top = max(top, m + s)
                if regime != BASE_REGIME:
                    ax.text(x, m + s + 0.8, f"{cell[(mkey, regime)]['delta']:+.1f}",
                            ha="center", va="bottom", fontsize=6.5,
                            color=style.INK_PRIMARY)

        ax.set_xticks(xs)
        ax.set_xticklabels([al.SPEC.label(m) for m in order])
        ax.set_ylabel(f"{al.DATASET} test accuracy (%)")
        ax.set_ylim(0, top * 1.16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(f"{al.DATASET}: integer delays, imposed at test vs at training",
                     fontweight="bold", fontsize=8.5, pad=3)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(handles=[Patch(facecolor=style.INK_MUTED, edgecolor="black",
                                 linewidth=0.6, alpha=REGIMES[g][1],
                                 hatch=REGIMES[g][2], label=REGIMES[g][0])
                           for g in regimes],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
                  frameon=False, handlelength=1.6, borderpad=0.15, labelspacing=0.3,
                  handletextpad=0.5)
        cu.save_fig(fig, os.path.join(out_dir, "round_compare.svg"))

    agg.to_csv(os.path.join(out_dir, "round_compare.csv"), index=False)
    print("\n" + agg.to_string(index=False))



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--rounding-root", default=ROUNDING_ROOT,
                    help=f"Sweep root of the trained-with-rounding runs "
                         f"(default {ROUNDING_ROOT}).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/AL/round_compare")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(al.OUT_ROOT, "round_compare")
    os.makedirs(out_dir, exist_ok=True)
    print(f"AL rounded-vs-unrounded comparison → {out_dir}\n"
          f"  families={FAMILIES}  seeds={args.seeds or al.SEEDS}  "
          f"rounding_root={args.rounding_root}")

    df = collect(args.rounding_root, args.exp_dir, args.seeds)
    plot_round_compare(df, out_dir)


if __name__ == "__main__":
    main()
