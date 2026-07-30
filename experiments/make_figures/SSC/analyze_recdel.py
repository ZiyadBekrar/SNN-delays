"""Aggregate multi-seed statistics for recurrent-delay SNNs trained on SSC.

Compares two model families (the hardcoded run set in ``SSC.config.RUN_DIRS``):
  SNN_recurrent_delays         (delays are learned)
  SNN_fixed_recurrent_delays   (delays frozen at init)

For each family it gathers every seed and produces test-accuracy, delay
distributions, delay-vs-weight scatter/trend, weight spectrum, signed
delay-weight relation, cross-seed KS stability and summary CSVs (see
``common.recdel`` for the individual figures).

Delay/weight statistics are read straight from the checkpoint state_dicts. Only
the optional test-set accuracy step rebuilds the model and runs a forward pass.

Usage
-----
    python experiments/make_figures/SSC/analyze_recdel.py
    python experiments/make_figures/SSC/analyze_recdel.py --ckpt best.pth --seeds 0 1 2 3 \
        --out-dir exp/SSC/make_figures/SSC_recdel
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve when run as
# ``python experiments/make_figures/SSC/analyze_recdel.py``.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd

import common.utils as cu
from common.recdel import run_recdel_analysis
import runs as ssc

# The recurrent kernel is centered at 1 + d (src/delrec/delay_layers_pytorch.py), so the
# physical applied delay is 1 + d steps (min 1). Report the physical delay.
PHYSICAL_DELAY_OFFSET_STEPS = 1.0

# Recurrent forward kernel used to rebuild every model for the test-set eval.
# Default 'triton_exact' (the persistent Triton exact scan). All kernels are
# numerically equivalent at σ=0 (see src/delrec/). Set 'v2' to force
# the portable pure-torch reference if Triton is unavailable, 'eventdriven' for the
# fast spike-sparse kernel, or None to let the layer pick its own default
# (eventdriven on CUDA, v2 on CPU). Overridable per-run with --forward-version.
FORWARD_VERSION = "triton_exact"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the RUN_DIRS paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all seeds in RUN_DIRS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to evaluate / read delays from.")
    ap.add_argument("--out-dir", default=os.path.join(ssc.OUT_ROOT, "SSC_recdel"))
    ap.add_argument("--no-test", action="store_true",
                    help="Skip rebuilding the model + SSC test-set forward pass.")
    ap.add_argument("--forward-version", default=FORWARD_VERSION,
                    help=f"Recurrent-delay forward kernel for eval (default "
                         f"{FORWARD_VERSION!r}).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    summary_rows = []
    for mkey in ssc.MODEL_PREFIXES:
        run_dirs = ssc.resolve_run_dirs(mkey, args.exp_dir, args.seeds)
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
            print(f"  {mkey:8s} seed {seed}: val(best)={best:.2f}%  ({os.path.basename(rd)})")
            summary_rows.append({"model": mkey, "seed": seed, "val_best_acc": best,
                                 "val_final_acc": float(curve[-1]), "run_dir": rd})
        results[mkey] = data

    # ---- verify the runs were trained the same way ----
    ssc.verify_runs(results, args.ckpt, args.out_dir)

    # ---- recompute accuracy on the SSC test set (real forward pass) ----
    if not args.no_test:
        print("\n=== Evaluating on the SSC test set (rebuilding models) ===")
        summary_by_key = {(r["model"], r["seed"]): r for r in summary_rows}
        for mkey, data in results.items():
            for seed in data["seeds"]:
                rd = data["run_dirs"][seed]
                try:
                    acc, loss, kernel = ssc.evaluate_test_accuracy(
                        mkey, rd, args.ckpt, forward_version_override=args.forward_version)
                except Exception as exc:  # missing GPU / spikingjelly / DCLS / dataset
                    print(f"[test] {mkey} seed {seed}: FAILED ({type(exc).__name__}: {exc})")
                    print("[test] falling back to validation-only reporting.")
                    for d in results.values():
                        d["test_accs"] = []
                    args.no_test = True
                    break
                data["test_accs"].append(acc)
                summary_by_key[(mkey, seed)]["test_acc"] = acc
                summary_by_key[(mkey, seed)]["test_loss"] = loss
                summary_by_key[(mkey, seed)]["eval_kernel"] = kernel
                print(f"  {mkey:8s} seed {seed}: test={acc:.2f}%  (loss={loss:.4f}, kernel={kernel})")
            if args.no_test:
                break

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "seed_summary.csv"), index=False)

    run_recdel_analysis(results, args.out_dir, ssc.SPEC, ssc.DATASET)


if __name__ == "__main__":
    main()
