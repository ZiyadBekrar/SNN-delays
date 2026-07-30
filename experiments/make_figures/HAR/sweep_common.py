"""Shared driver for the HAR input-perturbation sweeps.

Walks a sweep axis (family x swept value x seed), rebuilds each model once, and
hands it to a caller-supplied evaluate() for every perturbation magnitude, so
the subsampling, deletion and jitter sweeps share one loop.

The figure it draws compares learned against fixed delays at matched trained
delay spread rather than at matched initialization: training moves the delays, so
equal initial width would mean comparing different delay distributions.
"""

import os

import pandas as pd

from common import paths

from common import style
from common.perturb import aggregate, plot_magnitude_isoacc_overlay
import runs as har


def value_colors(axis):
    """One ordinal color per swept value (sequential ramp)."""
    return dict(zip(sorted(axis.values), style.ordinal_colors(len(axis.values))))


# The measured x axes the iso-accuracy overlay is re-plotted against, one twin each
# (see final_delay_coords): {stat: (filename suffix, x-axis label, title phrase)}.
# The stat keys index config.DELAY_STATS.
DELAY_STAT_AXES = {
    "std":  ("finalstd",  "final delay std (steps)",  "matched final delay std"),
    "mean": ("finalmean", "mean final delay (steps)", "matched mean final delay"),
    "max":  ("finalmax",  "max final delay (steps)",  "matched max final delay"),
}


def final_delay_coords(axis, key, order, stat, exp_dir=".", seeds=None, ckpt="best.pth"):
    """The x coordinates of one family on a trained delay axis: the ``stat`` ("std" / "mean"
    / "max") its runs actually ended with (mean over seeds), one coordinate per swept value
    in ``order``. The swept value is a knob.
    """
    coords = []
    for v in order:
        c = har.final_delay_stat(key, v, axis, stat=stat, exp_dir=exp_dir,
                                 seeds=seeds, ckpt=ckpt)
        if c is None:
            return None
        coords.append(c)
    return coords


# --------------------------------------------------------------------------- #
# Grid walk
# --------------------------------------------------------------------------- #
def run_model_sweep(evaluate, magnitudes, *, axis, n_perturb_seeds, clean_mag,
                    seeds, ckpt, seed_col="perturb_seed", build_model=None,
                    forward_version=None):
    """Evaluate ``evaluate`` over the (family × swept value × model-seed) grid of ``axis`` at
    every magnitude (× ``n_perturb_seeds`` realizations, except the clean magnitude which
    is deterministic and evaluated once).
    """
    build_model = build_model or har.build_test_model
    rows = []
    for mkey in axis.families:
        for val in axis.values:
            run_dirs = axis.resolve(mkey, val, ".", seeds)
            if not run_dirs:
                print(f"[warn] no runs for {mkey} {axis.value_dir(val)}")
                continue
            for seed in sorted(run_dirs):
                model, config, device = build_model(mkey, run_dirs[seed], ckpt,
                                                    forward_version=forward_version)
                loader = har.get_test_loader(config)
                print(f"\n{har.SPEC.label(mkey)} | {axis.value_dir(val)} | model seed {seed}")
                for mag in magnitudes:
                    n_ps = 1 if mag == clean_mag else n_perturb_seeds
                    for ps in range(n_ps):
                        acc = evaluate(model, config, device, loader, mag, ps)
                        rows.append({"model": mkey, axis.col: val, "magnitude": mag,
                                     "model_seed": seed, seed_col: ps, "acc": acc})
                        print(f"   mag={mag:>6g} pseed {ps}: acc={acc:.2f}%")
    if not rows:
        return None, None
    rows_df = pd.DataFrame(rows)
    summary = aggregate(rows, ["model", axis.col, "magnitude"])
    return rows_df, summary


def save_tables(rows_df, summary, out_dir, stem):
    """Persist the per-run and summary CSVs with a consistent naming scheme."""
    os.makedirs(out_dir, exist_ok=True)
    rows_df.to_csv(os.path.join(out_dir, f"{stem}_per_run.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, f"{stem}_summary.csv"), index=False)


# --------------------------------------------------------------------------- #
# Plotting (per-family + fixed-vs-learned comparison), axis- and label-agnostic
# --------------------------------------------------------------------------- #
# Learned and fixed models are compared at matched post-training delay spread
# rather than at matched initialization, because training moves the delays.
# DELAY_STAT_AXES also offers the delay mean and max. Those were exploratory.
PAPER_DELAY_AXIS = {"std": DELAY_STAT_AXES["std"]}


def load_summary(out_dir, stem, axis):
    """The sweep's summary table."""
    local = os.path.join(out_dir, f"{stem}_summary.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    shipped = paths.measurements("HAR", axis.out_prefix, stem, f"{stem}_summary.csv")
    if os.path.exists(shipped):
        print(f"  using the recorded measurement: {shipped}")
        return pd.read_csv(shipped)
    raise FileNotFoundError(
        f"no summary table for '{stem}'. Run the sweep without --plots-only "
        f"(needs a GPU and the dataset), or check {shipped}")


def make_sweep_plots(summary, out_dir, stem, mag_label, title_stem, *, axis,
                     baseline_mag=None, exp_dir=".", seeds=None):
    """Write the standard figure set from a ``summary`` DataFrame (columns
    model/<axis.col>/magnitude/mean/sem/n): * per family: accuracy heatmap over (value,
    magnitude) + accuracy-vs-magnitude lines (one per swept value) +
    relative-degradation-vs-magnitude lines (baseline = ``baseline_mag``, default the
    smallest magnitude). * per modality in ``axis.comparisons`` (empty for the wd sweep →
    skipped): 3-panel fixed|learned|Δ heatmap, iso-accuracy overlay, and one accuracy-vs-
    magnitude curve per value (learned solid / fixed dashed).
    """
    os.makedirs(out_dir, exist_ok=True)
    colors = value_colors(axis)
    labels = {v: f"{axis.key}={axis.pretty(v)}" for v in axis.values}
    order = sorted(axis.values)

    for modality, (fixed_key, learned_key) in axis.comparisons.items():
        fixed = summary[summary["model"] == fixed_key]
        learned = summary[summary["model"] == learned_key]
        if fixed.empty or learned.empty:
            continue
        # The iso-accuracy overlay interpolates a surface over the (value × magnitude)
        # grid, so it needs at least 2×2, skip it for degenerate (single-value or
        # single-magnitude) runs rather than let contourf raise.
        n_mags = fixed["magnitude"].nunique()
        if len(order) >= 2 and n_mags >= 2:
            # One twin per measured x: the same runs against a trained-delay statistic
            # each family actually converged to, both interpolated onto one common axis
            # so the two are compared at equal trained delays rather than at equal knob.
            for stat, (suffix, x_label, phrase) in PAPER_DELAY_AXIS.items():
                xf = final_delay_coords(axis, fixed_key, order, stat,
                                        exp_dir=exp_dir, seeds=seeds)
                xl = final_delay_coords(axis, learned_key, order, stat,
                                        exp_dir=exp_dir, seeds=seeds)
                if not (xf and xl):
                    continue
                plot_magnitude_isoacc_overlay(
                    fixed, learned,
                    os.path.join(out_dir,
                                 f"{stem}_isoacc_overlay_{modality}_vs_{suffix}.svg"),
                    title=f"{title_stem} — {modality}: iso-accuracy at {phrase}",
                    x_col=axis.col, x_order=order, x_label=x_label,
                    y_label=mag_label, x_coords=(xf, xl))


def print_summary(summary, title, axis):
    """Console dump of the summary table (mean ± SEM per model/value/magnitude)."""
    print(f"\n=== {title} (test acc, mean ± SEM over model×perturb seeds) ===")
    for _, r in summary.sort_values(["model", axis.col, "magnitude"]).iterrows():
        print(f"  {har.SPEC.label(r['model']):16s} {axis.key}{axis.pretty(r[axis.col]):<6} "
              f"mag={r['magnitude']:>7g}: {r['mean']:.2f} ± {r['sem']:.2f}%  (n={int(r['n'])})")
