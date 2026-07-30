"""Delay-reorganization-from-init analysis for recurrent-delay SNNs on HAR.

Compares each trained model's recurrent delays to the shared random initialization
saved at training time (``init_delays.npz``), quantifying how far learning moves the
delays, with the fixed family (frozen at init) as the built-in Δ≈0 control. Runs
once per swept value, writing one output subfolder per value (``.../HAR_reorg/
<value>/``). Works on either HAR sweep (``--sweep {std,wd}``). The wd sweep has only
the 2 learned families (no fixed Δ=0 control). All reads are from files, no rebuild.

Usage
-----
    python experiments/make_figures/HAR/analyze_reorg.py
    python experiments/make_figures/HAR/analyze_reorg.py --sweep wd --seeds 0 1 2
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np

import common.utils as cu
from common.reorg import run_reorg_analysis
import runs as har


def _load_init_delays(run_dir):
    """Per-layer init delay arrays (sorted by key), aligned with the order of
    ``load_recurrent_params``. None if the run has no init_delays.npz."""
    path = os.path.join(run_dir, "init_delays.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return [z[k] for k in sorted(z.files)]


def build_results(axis, value, exp_dir, seeds, ckpt):
    results = {}
    for mkey in axis.families:
        run_dirs = axis.resolve(mkey, value, exp_dir, seeds)
        if not run_dirs:
            print(f"[warn] no runs for {mkey} {axis.value_dir(value)}")
        data = {"seeds": [], "layers_per_seed": [], "init_per_seed": []}
        for seed in sorted(run_dirs):
            rd = run_dirs[seed]
            init = _load_init_delays(rd)
            if init is None:
                print(f"  [skip] {mkey} {axis.value_dir(value)} seed{seed}: no init_delays.npz")
                continue
            data["seeds"].append(seed)
            data["layers_per_seed"].append(cu.load_recurrent_params(rd, ckpt))
            data["init_per_seed"].append(init)
            print(f"  {mkey:12s} {axis.value_dir(value):<7} seed {seed}: loaded init + trained delays")
        results[mkey] = data
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--exp-dir", default=".")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/HAR_reorg")
    args = ap.parse_args()

    axis = har.get_axis(args)
    out_base = args.out_dir or axis.out_dir("HAR_reorg")
    for value in axis.values:
        out_dir = os.path.join(out_base, axis.value_dir(value))
        print(f"\n########## {axis.label} = {axis.pretty(value)} ##########")
        results = build_results(axis, value, args.exp_dir, args.seeds, args.ckpt)
        run_reorg_analysis(results, out_dir, har.SPEC)


if __name__ == "__main__":
    main()
