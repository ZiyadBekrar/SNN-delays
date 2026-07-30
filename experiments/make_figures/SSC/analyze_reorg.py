"""How far learning moves each delay from its random initialization, on SSC.

Uses ``common.reorg``, with the frozen fixed family as the exact zero-change control.

The SSC runs predate ``init_delays.npz``, so the initialization is reconstructed by
re-instantiating the model class on CPU: ``experiments/train.py --dataset ssc`` seeds immediately
before construction and nothing consumes the RNG in between. That reconstruction is
verified rather than assumed, since a fixed run never trains its delays, rebuilding it
must reproduce its stored delays exactly, and the script aborts if any run mismatches.
It is not valid on PS-MNIST, which builds its dataloaders in between.

Outputs:  figures/SSC/reorg/                                    [measurements]
Usage:    python experiments/make_figures/SSC/analyze_reorg.py [--seeds 0 1 2]
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np

import common.utils as cu
from common.reorg import run_reorg_analysis
from common.style import condition_of
import runs as ssc


def reconstruct_init(mkey, seed):
    """The delay draw ``experiments/train.py --dataset ssc`` would have produced for this (class, seed)."""
    import torch
    import delrec.networks as snn_mod
    from delrec.utils import seed_everything

    config = ssc.make_config()
    config.seed = seed
    seed_everything(seed=seed, is_cuda=False)
    with torch.no_grad():
        model = snn_mod.__dict__[ssc.MODEL_PREFIXES[mkey]](config)
        return [m.recurrent_delays.detach().cpu().numpy().copy()
                for m in model.modules() if hasattr(m, "recurrent_delays")]


def verify_reconstruction(exp_dir, seeds, ckpt):
    """Abort unless every fixed run's stored delays equal its rebuilt init."""
    checked = 0
    for mkey in [m for m in ssc.MODEL_PREFIXES if condition_of(m) == "fixed"]:
        for seed, rd in sorted(ssc.resolve_run_dirs(mkey, exp_dir, seeds).items()):
            stored = [np.asarray(l["delays"]) for l in
                      cu.load_recurrent_params(rd, ckpt, offset_steps=0.0)]
            rebuilt = reconstruct_init(mkey, seed)
            # The fixed classes round their delays in __init__. Compare on that basis.
            worst = max(float(np.abs(np.round(r) - s).max())
                        for s, r in zip(stored, rebuilt))
            if worst > 1e-5:
                raise SystemExit(
                    f"[abort] init reconstruction does not match the frozen delays of "
                    f"{mkey} seed {seed} (max |Δ| = {worst:g}). The RNG stream at model "
                    f"construction differs from training — reorg cannot be run without "
                    f"a stored init_delays.npz.")
            checked += 1
    print(f"[ok] init reconstruction verified exact on {checked} frozen-delay runs.")


def build_results(exp_dir, seeds, ckpt):
    results = {}
    for mkey in ssc.MODEL_PREFIXES:
        run_dirs = ssc.resolve_run_dirs(mkey, exp_dir, seeds)
        if not run_dirs:
            print(f"[warn] no runs for {mkey}")
        data = {"seeds": [], "layers_per_seed": [], "init_per_seed": []}
        for seed in sorted(run_dirs):
            rd = run_dirs[seed]
            data["seeds"].append(seed)
            data["layers_per_seed"].append(
                cu.load_recurrent_params(rd, ckpt, offset_steps=0.0))
            data["init_per_seed"].append(reconstruct_init(mkey, seed))
            print(f"  {mkey:12s} seed {seed}: trained delays + reconstructed init")
        results[mkey] = data
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--out-dir", default=os.path.join(ssc.OUT_ROOT, "reorg"))
    args = ap.parse_args()

    verify_reconstruction(args.exp_dir, args.seeds, args.ckpt)
    results = build_results(args.exp_dir, args.seeds, args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)
    run_reorg_analysis(results, args.out_dir, ssc.SPEC)
    print(f"\nFigures (SVG) + CSVs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
