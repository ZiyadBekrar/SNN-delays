"""Per-layer delay spread vs network depth for recurrent-delay SNNs on SSC.

Fits the empirical std of each recurrent layer's converged delay distribution and
plots it against layer depth (mean +/- SEM over seeds), for the four families in
``SSC.config.MODEL_PREFIXES``. The learned families carry the trend. The fixed
families are the flat reference. Tests whether the network builds a hierarchy of
temporal integration windows (deeper layers -> wider delays). See
``common.delay_depth``.

File-only: delays are read straight from the checkpoints, no forward pass.

Usage
-----
    python experiments/make_figures/SSC/analyze_delay_depth.py
    python experiments/make_figures/SSC/analyze_delay_depth.py --ckpt best.pth --seeds 0 1 2
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import common.utils as cu
from common.delay_depth import run_delay_depth_analysis
import runs as ssc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the RUN_DIRS paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all seeds in RUN_DIRS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to read delays from.")
    ap.add_argument("--out-dir", default=os.path.join(ssc.OUT_ROOT, "delay_depth"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for mkey in ssc.MODEL_PREFIXES:
        run_dirs = ssc.resolve_run_dirs(mkey, args.exp_dir, args.seeds)
        if not run_dirs:
            print(f"[warn] no runs found for {mkey} (seeds {args.seeds}) under {args.exp_dir}")
        data = {"seeds": [], "layers_per_seed": []}
        for seed in sorted(run_dirs):
            rd = run_dirs[seed]
            # offset_steps=0: std is invariant to the physical 1+d shift.
            data["seeds"].append(seed)
            data["layers_per_seed"].append(
                cu.load_recurrent_params(rd, args.ckpt, offset_steps=0.0))
            print(f"  {mkey:12s} seed {seed}: {os.path.basename(rd)}")
        results[mkey] = data

    run_delay_depth_analysis(results, args.out_dir, ssc.SPEC, ssc.DATASET)


if __name__ == "__main__":
    main()
