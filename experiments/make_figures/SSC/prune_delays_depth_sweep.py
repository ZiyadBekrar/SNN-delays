"""Layer-resolved branch pruning on SSC.

Asks whether the delays in a given layer carry classification signal. Per recurrent
layer, removes the branches whose delay is below (``short``) or above (``long``) a
threshold and re-evaluates the clean test set. The fixed random-delay family is the
null, its delays are unstructured, so pruning by threshold is arbitrary there. The
decisive read is the degradation-vs-layer summary: if pruning hurts the learned model
more, and does so in a depth-graded way, the delays are load-bearing.

Outputs:  figures/SSC/prune_delays_depth/{short,long}/    [GPU + SSC + checkpoints]
Usage:    python experiments/make_figures/SSC/prune_delays_depth_sweep.py [--directions long]
                 [--seeds 0] [--plots-only]
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.branch_prune import (evaluate_branch_pruned, pruned_fraction,
                                 DIRECTIONS, THRESHOLD_LABEL)
from common.delay_perturb import _recdel_modules
from common.perturb import (aggregate, plot_perf_vs_magnitude,
                            plot_degradation_vs_magnitude)
from common.prune_depth_plots import run_prune_depth_plots
from common.utils import save_fig
from common.style import type_of
import runs as ssc

from delrec.utils import calc_metric_SSC

# ======================= TUNE THE SWEEP HERE ============================== #
# Delay thresholds d (timesteps). 'long' removes delay > d (largest d removes
# least = identity). 'short' removes delay < d (d=0 removes nothing = identity).
# The largest value should exceed every model's max recurrent delay (SSC max ~54).
D_THRESHOLDS = [0, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 32, 48, 64]
# Threshold used for each direction's degradation-vs-layer summary figure (must be
# in D_THRESHOLDS). 'long'>16 removes the extended tail. 'short'<8 removes the
# low-delay core.
SUMMARY_D = {"long": 16, "short": 8}
MODEL_SEEDS = None
CKPT = "best.pth"
# Recurrent forward kernel used to rebuild every model for eval (see analyze_recdel).
FORWARD_VERSION = "triton_exact"
# ========================================================================= #


def _baseline_d(direction):
    """Identity threshold (removes least): d=0 for 'short', max d for 'long'."""
    return 0 if direction == "short" else max(D_THRESHOLDS)


def _collect_rows(directions):
    """Sweep every (family, seed, direction, layer, threshold). Return per-run rows."""
    rows = []
    # The SSC sparse .h5 decode is ~90s per full test-set pass and dominates each eval
    # (the Triton forward itself is ~6s). The test inputs are identical across every
    # family/seed/prune, so decode the loader ONCE and reuse the materialised batches.
    cached_loader = None
    for mkey in ssc.MODEL_PREFIXES:
        run_dirs = ssc.resolve_run_dirs(mkey, ".", MODEL_SEEDS)
        if not run_dirs:
            print(f"[warn] no runs for {mkey}")
            continue
        for seed in sorted(run_dirs):
            model, config, device, kernel = ssc.build_test_model(
                mkey, run_dirs[seed], CKPT, forward_version_override=FORWARD_VERSION)
            if cached_loader is None:
                print("Caching SSC test set (one-time decode)...", flush=True)
                cached_loader = list(ssc.get_test_loader(config))
            loader = cached_loader
            n_layers = len(_recdel_modules(model))
            print(f"\n{ssc.SPEC.label(mkey)} | seed {seed} | {n_layers} recurrent layers | kernel={kernel}")
            for direction in directions:
                for layer_index in range(n_layers):
                    for d in D_THRESHOLDS:
                        acc = evaluate_branch_pruned(loader, model, device, calc_metric_SSC,
                                                     direction, d, layer_index=layer_index)
                        fp = pruned_fraction(model, direction, d, layer_index=layer_index)
                        rows.append({"model": mkey, "direction": direction,
                                     "layer": layer_index + 1, "magnitude": d,
                                     "model_seed": seed, "pruned_frac": fp, "acc": acc})
                        print(f"   {direction:5s} L{layer_index+1} d={d:>3g} "
                              f"({100*fp:4.1f}% pruned): acc={acc:.2f}%")
    return rows


def _type_groups(dir_rows):
    """Ordered MODEL_PREFIXES keys grouped by delay type, keeping only types that have
    runs present in ``dir_rows``, so axonal and synaptic each get their own figures."""
    present = {r["model"] for r in dir_rows}
    groups = {}
    for k in ssc.MODEL_PREFIXES:
        if k in present:
            groups.setdefault(type_of(k), []).append(k)
    return groups


def _plot_degradation_vs_layer(rows, out_dir, direction, order):
    """Headline: relative degradation at SUMMARY_D[direction] vs layer, per family."""
    df = pd.DataFrame([r for r in rows if r["direction"] == direction
                       and r["model"] in order])
    base_d, sum_d = _baseline_d(direction), SUMMARY_D[direction]
    base = (df[df["magnitude"] == base_d]
            .set_index(["model", "layer", "model_seed"])["acc"])
    cur = df[df["magnitude"] == sum_d].copy()
    cur["degr"] = cur.apply(
        lambda r: 100.0 * (base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"])
        / base.loc[(r["model"], r["layer"], r["model_seed"])], axis=1)
    g = (cur.groupby(["model", "layer"])["degr"]
         .agg(mean="mean", sd="std", n="count").reset_index())
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    g.to_csv(os.path.join(out_dir, "degradation_vs_layer.csv"), index=False)

    comp = "<" if direction == "short" else ">"
    layers = sorted(g["layer"].unique())
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mkey in [m for m in order if m in g["model"].values]:
        sub = g[g["model"] == mkey].set_index("layer")
        means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
        sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
        ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                    color=ssc.SPEC.color(mkey), ls=ssc.SPEC.linestyle(mkey),
                    label=ssc.SPEC.label(mkey))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Recurrent layer (depth)")
    ax.set_ylabel(f"Relative degradation (%) — prune delay {comp} {sum_d}")
    ax.set_title(f"SSC: {direction}-branch prune sensitivity vs depth (mean ± SEM over seeds)")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "degradation_vs_layer.svg"))


def run_direction(rows, direction, out_root):
    """Write per-layer + summary figures for one prune direction, split by delay type
    (axonal-only / synaptic-only) into ``<direction>/<type>/`` subtrees."""
    dir_rows = [r for r in rows if r["direction"] == direction]
    for dtype, order in _type_groups(dir_rows).items():
        type_rows = [r for r in dir_rows if r["model"] in order]
        dir_root = os.path.join(out_root, direction, dtype)
        for layer in sorted({r["layer"] for r in type_rows}):
            out_dir = os.path.join(dir_root, f"layer{layer}")
            os.makedirs(out_dir, exist_ok=True)
            layer_rows = [r for r in type_rows if r["layer"] == layer]
            summary = aggregate(layer_rows, ["model", "magnitude"])
            summary.to_csv(os.path.join(out_dir, f"prune_{direction}_summary.csv"), index=False)
            title = f"SSC {dtype} L{layer} — {DIRECTIONS[direction]}"
            plot_perf_vs_magnitude(
                summary, os.path.join(out_dir, f"prune_{direction}_perf_vs_magnitude.svg"),
                title=f"{title}: accuracy", hue="model", order=order,
                labels=ssc.SPEC.labels, colors=ssc.SPEC.colors, xlabel=THRESHOLD_LABEL,
                linestyles=ssc.SPEC.linestyles)
            plot_degradation_vs_magnitude(
                summary, os.path.join(out_dir, f"prune_{direction}_degradation_vs_magnitude.svg"),
                title=f"{title}: degradation", hue="model", order=order,
                labels=ssc.SPEC.labels, colors=ssc.SPEC.colors, xlabel=THRESHOLD_LABEL,
                baseline_mag=_baseline_d(direction), linestyles=ssc.SPEC.linestyles)
        os.makedirs(dir_root, exist_ok=True)
        _plot_degradation_vs_layer(dir_rows, dir_root, direction, order)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=os.path.join(ssc.OUT_ROOT, "prune_delays_depth"))
    ap.add_argument("--directions", nargs="+", default=list(DIRECTIONS),
                    choices=list(DIRECTIONS), help="Which prune directions to run.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these model seeds (default: all in RUN_DIRS).")
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip pruning, reload prune_depth_per_run.csv and re-render figures.")
    args = ap.parse_args()
    global MODEL_SEEDS
    if args.seeds is not None:
        MODEL_SEEDS = args.seeds
    os.makedirs(args.out_root, exist_ok=True)

    per_run_csv = os.path.join(args.out_root, "prune_depth_per_run.csv")
    if args.plots_only:
        if not os.path.exists(per_run_csv):
            print(f"[error] --plots-only but no cached results at {per_run_csv}")
            return
        rows = pd.read_csv(per_run_csv).to_dict("records")
    else:
        rows = _collect_rows(args.directions)
        if not rows:
            print("[error] no results.")
            return
        pd.DataFrame(rows).to_csv(per_run_csv, index=False)
    present = {r["direction"] for r in rows}
    for direction in [d for d in args.directions if d in present]:
        run_direction(rows, direction, args.out_root)
    run_prune_depth_plots(rows, args.out_root, ssc.SPEC, "SSC",
                          os.path.join(ssc.OUT_ROOT, "delay_depth",
                                       "delay_depth_std_summary.csv"))
    print(f"\nFigures (SVG) + CSVs written to {args.out_root}/")


if __name__ == "__main__":
    main()
