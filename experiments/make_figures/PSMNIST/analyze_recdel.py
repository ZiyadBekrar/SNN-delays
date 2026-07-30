"""Aggregate multi-seed statistics for recurrent-delay SNNs trained on PSMNIST.

The PSMNIST twin of ``experiments/make_figures/SSC/analyze_recdel.py``. Compares the four families
in ``PSMNIST.config.MODEL_PREFIXES``, axonal & synaptic recurrent delays, each
learned & fixed, and produces test-accuracy, delay distributions, delay-vs-weight
scatter/trend, weight spectrum, signed delay-weight, cross-seed KS stability and
summary CSVs (see ``common.recdel``).

Delay/weight statistics are read straight from the checkpoint state_dicts. The test
accuracy is read from each run's ``final_test.json`` (cached by
``config.get_test_accuracy``, ``experiments/train.py --dataset psmnist`` prints it but never writes it, so
the first evaluation rebuilds the model + forward pass. Use ``--recompute`` to force
re-evaluation). After the first pass it reads recorded measurements only (no GPU / dataset).

Usage
-----
    python experiments/make_figures/PSMNIST/analyze_recdel.py
    python experiments/make_figures/PSMNIST/analyze_recdel.py --ckpt best.pth --seeds 0 1 2 3 4
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd

import common.utils as cu
from common.recdel import run_recdel_analysis
import runs as psm

# The recurrent kernel is centered at 1 + d (src/delrec/), so the
# physical applied delay is 1 + d steps (min 1). Report the physical delay.
PHYSICAL_DELAY_OFFSET_STEPS = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all of config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to evaluate / read delays from.")
    ap.add_argument("--out-dir", default=os.path.join(psm.OUT_ROOT, "PSMNIST_recdel"))
    ap.add_argument("--no-test", action="store_true",
                    help="Skip the test-set accuracy (read/compute), report validation only.")
    ap.add_argument("--recompute", action="store_true",
                    help="Force re-evaluation of the test accuracy (rebuild + forward pass) "
                         "instead of reading each run's final_test.json.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    summary_rows = []
    for mkey in psm.MODEL_PREFIXES:
        run_dirs = psm.resolve_run_dirs(mkey, args.exp_dir, args.seeds)
        if not run_dirs:
            print(f"[warn] no runs found for {mkey} (seeds {args.seeds}) under {args.exp_dir}")
        data = {"best_accs": [], "test_accs": [], "curves": [], "layers_per_seed": [],
                "seeds": [], "run_dirs": run_dirs}
        for seed in sorted(run_dirs):
            rd = run_dirs[seed]
            curve = cu.load_val_accuracy(rd)
            best = cu.best_acc_from_ckpt(rd, args.ckpt)
            if best is None:
                best = float(np.max(curve))
            data["seeds"].append(seed)
            data["curves"].append(curve)
            data["best_accs"].append(best)
            data["layers_per_seed"].append(
                cu.load_recurrent_params(rd, args.ckpt, offset_steps=PHYSICAL_DELAY_OFFSET_STEPS))
            print(f"  {mkey:12s} seed {seed}: val(best)={best:.2f}%  ({os.path.basename(rd)})")
            summary_rows.append({"model": mkey, "seed": seed, "val_best_acc": best,
                                 "val_final_acc": float(curve[-1]), "run_dir": rd})
        results[mkey] = data

    # ---- test-set accuracy (cached in final_test.json, or recomputed) ----
    if not args.no_test:
        print("\n=== PSMNIST test accuracy (from final_test.json, --recompute to re-evaluate) ===")
        summary_by_key = {(r["model"], r["seed"]): r for r in summary_rows}
        for mkey, data in results.items():
            for seed in data["seeds"]:
                rd = data["run_dirs"][seed]
                try:
                    acc, from_cache = psm.get_test_accuracy(
                        mkey, rd, args.ckpt, recompute=args.recompute)
                except Exception as exc:  # missing GPU / spikingjelly / DCLS / dataset
                    print(f"[test] {mkey} seed {seed}: FAILED ({type(exc).__name__}: {exc})")
                    print("[test] falling back to validation-only reporting.")
                    for d in results.values():
                        d["test_accs"] = []
                    args.no_test = True
                    break
                data["test_accs"].append(acc)
                summary_by_key[(mkey, seed)]["test_acc"] = acc
                src = "cached" if from_cache else "computed"
                print(f"  {mkey:12s} seed {seed}: test={acc:.2f}%  ({src})")
            if args.no_test:
                break

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "seed_summary.csv"), index=False)

    run_recdel_analysis(results, args.out_dir, psm.SPEC, psm.DATASET)


if __name__ == "__main__":
    main()
