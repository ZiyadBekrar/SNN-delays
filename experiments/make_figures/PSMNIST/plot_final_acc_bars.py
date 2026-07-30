"""Final PS-MNIST test accuracy, one bar per model family (mean and std over seeds).

Thin main over ``common.accbars``, everything PS-MNIST-specific lives in
``experiments/make_figures/PSMNIST/runs.py``. Accuracy is measured under each run's own pixel
permutation, at sigma=0 with rounded delays.

Outputs:  figures/PSMNIST/final_acc_bars/
Usage:    python experiments/make_figures/PSMNIST/plot_final_acc_bars.py [--seeds 0 1 2]
                 [--recompute]
"""

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

from common.accbars import run_accbars_analysis
import runs as psmnist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to evaluate (default best.pth, as perf_PSMNIST does).")
    ap.add_argument("--recompute", action="store_true",
                    help="Re-run the test-set evaluation even if final_test.json exists.")
    ap.add_argument("--forward-version", default=psmnist.FORWARD_VERSION,
                    help=f"Recurrent kernel for the eval (default "
                         f"{psmnist.FORWARD_VERSION!r}; synaptic families map it to "
                         f"'eventdriven').")
    ap.add_argument("--out-dir", default=os.path.join(psmnist.OUT_ROOT, "final_acc_bars"))
    args = ap.parse_args()

    print(f"PSMNIST final test accuracy → {args.out_dir}\n"
          f"  seeds={args.seeds or psmnist.SEEDS}  ckpt={args.ckpt}  "
          f"kernel={args.forward_version}")
    run_accbars_analysis(psmnist, args.out_dir, exp_dir=args.exp_dir, seeds=args.seeds,
                         ckpt=args.ckpt, recompute=args.recompute,
                         forward_version=args.forward_version)


if __name__ == "__main__":
    main()
