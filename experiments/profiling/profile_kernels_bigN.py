"""Fill in the axonal ``triton_exact`` forward+backward points at N >= 512.

``profile_kernels.py`` records them as N/A: the backward kernel tiles into shared
memory and at N=512 asks for 160 KiB against the A40's 99 KiB, so Triton raises
``OutOfResources`` before launching. The dominant term is the W tile of the
``gz = gr @ W`` dot, whose width nothing in the maths requires, so this script re-runs
those points with the largest ``BK`` (64, then 32, then 16) that fits and writes them
to ``<run>/axonal/kernel_profile_bigN.csv`` with a ``bk`` column. The main CSV is left
untouched. ``BLOCK_B`` cannot be lowered instead, ``tl.dot`` needs all tile dims >= 16.

Usage:    CUDA_VISIBLE_DEVICES=1 python experiments/profiling/profile_kernels_bigN.py
"""

from pathlib import Path

import pandas as pd
import torch

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

from common import paths                     # noqa: E402  (needs the paths above)

import delrec.triton_kernels.recdel_base as rb
from profile_kernels import (DEFAULTS, SEED, bench, build_layer, make_input)
from delrec.delay_layers import axonal_recdel

def _latest_sweep_run():
    """Newest profiling sweep under trained_models/profiling/, or None if there is none.

    The measurements are hardware-specific, so this fills in whichever run is
    present rather than naming one machine's timestamped directory."""
    import glob
    runs = sorted(glob.glob(paths.runs("profiling", "sweeps_*")))
    return Path(runs[-1]) if runs else None


SWEEP_RUN = _latest_sweep_run()
FILL_N = [512, 1024]                 # the two N-sweep points with no fwdbwd row
BK_CANDIDATES = [64, 32, 16]         # W-column tile. 16 is tl.dot's minimum

_backward = rb.recdel_triton_backward


def _patch_bk(bk):
    """Point the autograd Function's backward at a narrower W-column tile."""
    rb.recdel_triton_backward = lambda *a, **kw: _backward(*a, **{**kw, "bk": bk})


def main():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")
    T, B, D, sigma = (DEFAULTS[k] for k in ("T", "B", "D", "sigma"))

    rows = []
    for N in FILL_N:
        layer = build_layer(axonal_recdel, N, D, sigma, device)
        x = make_input(T, B, N, device)
        for bk in BK_CANDIDATES:
            _patch_bk(bk)
            # `bench` swallows OOM / Triton OutOfResources and returns NaN.
            t_ms, mem_mib, fr = bench(layer, x, "triton_exact", True, device)
            if t_ms != t_ms:
                print(f"N={N:5d} BK={bk:2d}: out of shared memory")
                continue
            print(f"N={N:5d} BK={bk:2d}: {t_ms:8.1f} ms, {mem_mib:8.0f} MiB")
            rows.append(dict(kind="axonal", sweep="N", value=float(N), T=T, B=B, N=N,
                             D=D, sigma=sigma, regime="fwdbwd", mode="triton_exact",
                             time_ms=t_ms, mem_mib=mem_mib, firing=fr, bk=bk))
            break
        else:
            print(f"N={N}: no BK fits, not recorded")
        del layer, x
        torch.cuda.empty_cache()

    rb.recdel_triton_backward = _backward
    out = SWEEP_RUN / "axonal" / "kernel_profile_bigN.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
