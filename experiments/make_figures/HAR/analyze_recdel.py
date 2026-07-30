"""Aggregate multi-seed delay/weight statistics for recurrent-delay SNNs on HAR.

Runs the SSC-equivalent recdel analysis once per swept value, writing one output
subfolder per value (``.../HAR_recdel/<value>/``). Works on either HAR sweep
(``--sweep {std,wd}``): the delay-std-init sweep (4 families, axonal & synaptic,
each learned & fixed) or the weight-decay sweep (the 2 learned families only). Delay/
weight statistics are read from the checkpoint state_dicts. Test accuracy from each
run's final_test.json, no model rebuild.

Usage
-----
    python experiments/make_figures/HAR/analyze_recdel.py
    python experiments/make_figures/HAR/analyze_recdel.py --sweep wd --values 0 0.01 --no-test
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd

import common.utils as cu
from common.recdel import run_recdel_analysis
import runs as har


def build_results(axis, value, exp_dir, seeds, ckpt, use_test):
    results = {}
    summary_rows = []
    for mkey in axis.families:
        run_dirs = axis.resolve(mkey, value, exp_dir, seeds)
        if not run_dirs:
            print(f"[warn] no runs for {mkey} {axis.value_dir(value)} (seeds {seeds}) under {exp_dir}")
        data = {"best_accs": [], "test_accs": [], "curves": [], "layers_per_seed": [],
                "seeds": [], "run_dirs": run_dirs}
        for seed in sorted(run_dirs):
            rd = run_dirs[seed]
            curve = cu.load_val_accuracy(rd)
            best = cu.best_acc_from_ckpt(rd, ckpt)
            if best is None:
                best = float(np.max(curve))
            data["seeds"].append(seed)
            data["curves"].append(curve)
            data["best_accs"].append(best)
            data["layers_per_seed"].append(cu.load_recurrent_params(rd, ckpt))
            row = {"model": mkey, axis.col: value, "seed": seed, "val_best_acc": best,
                   "val_final_acc": float(curve[-1]), "run_dir": rd}
            if use_test:
                test = har.read_test_accuracy(rd)
                if test is not None:
                    data["test_accs"].append(test)
                    row["test_acc"] = test
            print(f"  {mkey:12s} {axis.value_dir(value):<7} seed {seed}: val(best)={best:.2f}%  "
                  f"({os.path.basename(rd)})")
            summary_rows.append(row)
        results[mkey] = data
    return results, summary_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth", help="Checkpoint to read delays from.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/HAR_recdel")
    ap.add_argument("--no-test", action="store_true",
                    help="Skip reading final_test.json, report checkpoint val accuracy only.")
    args = ap.parse_args()

    axis = har.get_axis(args)
    out_base = args.out_dir or axis.out_dir("HAR_recdel")
    all_summary = []
    for value in axis.values:
        out_dir = os.path.join(out_base, axis.value_dir(value))
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n########## {axis.label} = {axis.pretty(value)} ##########")
        results, summary_rows = build_results(axis, value, args.exp_dir, args.seeds,
                                              args.ckpt, not args.no_test)
        all_summary.extend(summary_rows)
        run_recdel_analysis(results, out_dir, har.SPEC, f"HAR ({axis.key}={axis.pretty(value)})")

    if all_summary:
        pd.DataFrame(all_summary).to_csv(
            os.path.join(out_base, "seed_summary.csv"), index=False)
        print(f"\nCombined per-run summary: {os.path.join(out_base, 'seed_summary.csv')}")


if __name__ == "__main__":
    main()
