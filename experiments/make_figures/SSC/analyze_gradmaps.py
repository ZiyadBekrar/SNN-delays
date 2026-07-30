"""Input-gradient temporal maps for recurrent-delay SNNs trained on SSC.

It runs on the hardcoded recurrent models shared with ``analyze_recdel.py``
(``SSC.config.RUN_DIRS``): learned vs fixed delays, all seeds, plus the
feedforward/DCLS models listed in ``EXTRA_RUNS``.

Usage
-----
    python experiments/make_figures/SSC/analyze_gradmaps.py
    python experiments/make_figures/SSC/analyze_gradmaps.py --T 3000 --t-crop 2850 --classes 0 1 2
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import common.utils as cu
from common import style
from common import paths
from common.gradmaps import (
    load_gradmap_model, compute_gradmaps_batched, crop_gradmaps,
    plot_gradmap_paper_row, plot_gradmap_energy_profiles,
)
import runs as ssc

from delrec.utils import seed_everything


# Recurrent forward kernel used to rebuild every model for the gradient maps.
# Default 'triton_exact' (the persistent Triton exact scan). All kernels are
# numerically equivalent (spikes bit-identical, grads to fp32 tolerance. See
# src/delrec/). Set 'v2' to force the portable pure-torch reference if
# Triton is unavailable, 'eventdriven' for the fast spike-sparse kernel, or None to
# let the layer pick its own default (eventdriven on CUDA, v2 on CPU).
FORWARD_VERSION = "triton_exact"

# Extra runs appended to the hardcoded recurrent set. Each entry needs
# {"model", "path", "tag", "seed"}. "dtype" (the per-seed figure's filename) and
# "cond" (its panel title) default to "tag".
# The feedforward/DCLS model is a delay type of its own, no recurrence at all,
# so it gets its own single-panel feedforward.svg rather than joining the
# axonal/synaptic rows. Only seeds 0-2 were trained.
EXTRA_RUNS = [
    {"model": "SNN_axonal_feedforward_delays",
     "path": paths.runs(f"SSC/SNN_feedforward_axonal_delays/SSC_DCLS_{s}"),
     "tag": "Feedforward delays", "seed": s,
     "dtype": "feedforward", "cond": "Feedforward"}
    for s in (0, 1, 2)
]

_plot_executor = ThreadPoolExecutor(max_workers=4)


def build_runs(seeds=None, untrained=True):
    """Build the run list from RUN_DIRS (all families, axonal & synaptic, learned & fixed, all
    seeds), optionally one untrained model per delay type and seed (same architecture at
    its random init, no checkpoint, the control), then EXTRA_RUNS.
    """
    runs = []
    for mkey in ssc.MODEL_PREFIXES:
        run_dirs = ssc.resolve_run_dirs(mkey, ".", seeds)
        for seed in sorted(run_dirs):
            runs.append({"model": ssc.MODEL_PREFIXES[mkey], "path": run_dirs[seed],
                         "key": run_dirs[seed], "tag": ssc.SPEC.label(mkey),
                         "seed": seed, "mkey": mkey, "dtype": style.type_of(mkey),
                         "cond": style.condition_of(mkey).capitalize()})
    if untrained:
        for dtype, mkey in (("axonal", "ax_learned"), ("synaptic", "syn_learned")):
            for seed in sorted({r["seed"] for r in runs if r["dtype"] == dtype}):
                runs.append({"model": ssc.MODEL_PREFIXES[mkey], "path": None,
                             "key": f"untrained/{dtype}/seed{seed}",
                             "tag": f"{dtype.capitalize()} untrained", "seed": seed,
                             "mkey": mkey, "dtype": dtype, "cond": "Untrained"})
    runs += [r for r in EXTRA_RUNS if seeds is None or r["seed"] in seeds]
    for r in runs:   # EXTRA_RUNS entries carry only the four original fields
        r.setdefault("key", r["path"])
        r.setdefault("dtype", r["tag"])
        r.setdefault("cond", r["tag"])
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all seeds in RUN_DIRS.")
    ap.add_argument("--T", type=int, nargs="+", default=[3000],
                    help="Sequence length(s) to run gradmaps for.")
    # The full catalog by default: one map per (class, seed, model). The paper
    # illustrates a single class but states that the maps for every class and seed
    # are available here, so this is what a plain run produces. Narrow it with
    # `--classes 4` when you only want the illustrated one.
    ap.add_argument("--classes", type=int, nargs="+", default=list(range(35)),
                    help="Target class indices (default: all 35).")
    ap.add_argument("--no-untrained", action="store_true",
                    help="Drop the untrained (random-init) control panel.")
    ap.add_argument("--loss-point", default="final", choices=["final", "sum"])
    ap.add_argument("--t-crop", type=int, default=2850,
                    help="Crop gradmaps to [t_crop:] before energy plots (0 = no crop).")
    ap.add_argument("--out-dir", default=str(Path(ssc.OUT_ROOT) / "gradmaps"))
    args = ap.parse_args()

    device = cu.get_device()

    target_classes = args.classes
    runs = build_runs(args.seeds, untrained=not args.no_untrained)
    if not runs:
        print("[error] no runs resolved from RUN_DIRS, nothing to do.")
        return

    models = {}
    for r in runs:
        # Synaptic families have no triton_exact path (would fall through to the slow
        # pure-torch v2). Kernel_for routes them to eventdriven. Speed only.
        fv = cu.kernel_for(r["mkey"], FORWARD_VERSION, device) if "mkey" in r else FORWARD_VERSION
        cfg = ssc.make_config()
        if r["path"] is None:   # untrained control: seed its random init per seed
            seed_everything(seed=r["seed"], is_cuda=True)
            cfg.seed = r["seed"]
        models[r["key"]] = load_gradmap_model(r["model"], r["path"], args.ckpt,
                                              device, cfg, forward_version=fv)

    first_per_tag = {}
    for r in runs:
        first_per_tag.setdefault(r["tag"], r)
    unique_tags = list(first_per_tag.keys())

    for T in args.T:
        out_dir = Path(args.out_dir) / f"T{T}"

        all_gradmaps_by_path = {}
        for r in runs:
            model, cfg = models[r["key"]]
            all_gradmaps_by_path[r["key"]] = compute_gradmaps_batched(
                model, cfg, T, device, target_classes, mode='zeros', loss_point=args.loss_point,
            )

        if args.t_crop > 0:
            all_gradmaps_by_path = {p: crop_gradmaps(gm, args.t_crop)
                                    for p, gm in all_gradmaps_by_path.items()}

        all_gradmaps_per_seed = {
            tag: {
                cls: [all_gradmaps_by_path[r["key"]][cls] for r in runs if r["tag"] == tag]
                for cls in target_classes
            }
            for tag in unique_tags
        }

        # Raw energy and its Frobenius-normalized twin: the raw profiles compare
        # magnitudes across models, the normalized ones compare the shape of the
        # temporal profile once each model's overall gradient scale is divided out.
        for fig_m, path_m in plot_gradmap_energy_profiles(all_gradmaps_per_seed, target_classes, out_dir):
            _plot_executor.submit(cu.save_fig, fig_m, path_m)
        for fig_m, path_m in plot_gradmap_energy_profiles(all_gradmaps_per_seed, target_classes,
                                                          out_dir, normalize=True):
            _plot_executor.submit(cu.save_fig, fig_m, path_m)

        # One figure per (class, seed, delay type): learned / fixed / untrained of
        # that type, on a single shared color scale so the panels are directly
        # comparable. Written as class<c>/seed<s>/<dtype>.svg.
        groups = {}
        for r in runs:
            groups.setdefault((r["dtype"], r["seed"]), []).append(r)

        panel_order = {"Learned": 0, "Fixed": 1, "Untrained": 2}
        for (dtype, seed), group in sorted(groups.items()):
            group = sorted(group, key=lambda r: panel_order.get(r["cond"], 99))
            futures = []
            for target_class in target_classes:
                grads_list = [all_gradmaps_by_path[r["key"]][target_class] for r in group]
                # Panel titles carry the condition only. The delay type, class and
                # seed are in the path (and belong in the paper's caption).
                labels = [r["cond"] for r in group]
                # The untrained control's gradient is ~10^3x smaller than the
                # trained ones, so it gets its own scale + colorbar (blank on the
                # shared one). Learned vs fixed stay strictly comparable.
                own = [r["cond"] == "Untrained" for r in group]
                fig = plot_gradmap_paper_row(grads_list, labels, own_scale=own)
                futures.append(_plot_executor.submit(
                    cu.save_fig, fig,
                    out_dir / "gradmaps_per_seed" /
                    f"class{target_class}" / f"seed{seed}" / f"{dtype}.svg"))
            for f in futures:   # drain before building the next group's figures
                f.result()

    _plot_executor.shutdown(wait=True)
    print(f"\nGradmap figures (SVG) written under {args.out_dir}/")


if __name__ == "__main__":
    main()
