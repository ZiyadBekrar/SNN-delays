"""Spiking-efficiency + per-class analysis for recurrent-delay SNNs on HAR.

One forward pass per run over the test set yields (test accuracy, hidden firing
rate, per-class accuracy). From these the analysis writes an accuracy-vs-firing
scatter + firing-rate bar (energy view) and, when the sweep has fixed runs to
compare against, per-modality per-class Δaccuracy bars (learned − fixed). Runs over
either HAR sweep (``--sweep {std,wd}``). Rebuilds each model in the σ=0 eval regime
(GPU + dataset needed). Outputs nest under figures/HAR/<sweep>/HAR_efficiency/.

Usage
-----
    python experiments/make_figures/HAR/analyze_efficiency.py
    python experiments/make_figures/HAR/analyze_efficiency.py --sweep wd --seeds 0 1 2
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import pandas as pd

from common.efficiency import (evaluate_efficiency, run_efficiency_analysis,
                               run_per_class_analysis)
import runs as har

CKPT = "best.pth"
# Recurrent forward kernel for all model rebuilds. Default 'triton_exact'. All
# kernels are numerically equivalent (see src/delrec/). Alternatives:
# 'v2' (portable pure-torch), 'eventdriven' (fast spike-sparse), None (layer default).
FORWARD_VERSION = "triton_exact"


def run(out_dir, axis, seeds):
    run_rows, class_rows = [], []
    for mkey in axis.families:
        for value in axis.values:
            run_dirs = axis.resolve(mkey, value, ".", seeds)
            if not run_dirs:
                print(f"[warn] no runs for {mkey} {axis.value_dir(value)}")
                continue
            for seed in sorted(run_dirs):
                model, config, device = har.build_test_model(
                    mkey, run_dirs[seed], CKPT, forward_version=FORWARD_VERSION)
                loader = har.get_test_loader(config)
                acc, fr, per_class = evaluate_efficiency(loader, model, device,
                                                         config.output_size)
                print(f"{har.SPEC.label(mkey):16s} {axis.value_dir(value):<7} seed{seed}: "
                      f"acc={acc:.2f}%  firing={fr:.4f}")
                run_rows.append({"model": mkey, axis.col: value, "seed": seed,
                                 "acc": acc, "firing_rate": fr})
                for c, a in enumerate(per_class):
                    class_rows.append({"model": mkey, axis.col: value, "seed": seed,
                                       "class": c, "acc": a})
    if not run_rows:
        print("[error] no results.")
        return
    run_efficiency_analysis(pd.DataFrame(run_rows), out_dir, har.SPEC)
    # Per-class Δ(learned−fixed) needs fixed runs, only the std sweep has them.
    if axis.comparisons:
        run_per_class_analysis(pd.DataFrame(class_rows), out_dir, har.SPEC, axis.comparisons)
    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/HAR_efficiency")
    args = ap.parse_args()
    axis = har.get_axis(args)
    out_dir = args.out_dir or axis.out_dir("HAR_efficiency")
    run(out_dir, axis, args.seeds)


if __name__ == "__main__":
    main()
