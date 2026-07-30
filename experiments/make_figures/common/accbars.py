"""Final test accuracy as one bar per model family (mean ± std over seeds).

The headline accuracy figure for the datasets whose runs are a flat family × seed
grid with no swept variable, AL, PSMNIST (contrast HAR, where the same quantity is a
line per family against the sweep: ``experiments/make_figures/HAR/plot_final_acc_compare.py``).

The dataset glue module (``experiments/make_figures/<dataset>/runs.py``) supplies everything
dataset-specific: ``MODEL_PREFIXES`` (the families, in plot order), ``SPEC``,
``DATASET``, ``resolve_run_dirs(mkey, exp_dir, seeds)`` and
``get_test_accuracy(mkey, run_dir, ckpt, recompute)``, the latter caching its result
in the run's ``final_test.json`` so only the first pass needs a GPU.

Style follows the repo's chart (``experiments/make_figures/common/style.py``): color = delay type
(axonal purple / synaptic red), hatch = condition (learned plain, fixed hatched).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style


def collect(cfg, exp_dir=".", seeds=None, ckpt="best.pth", recompute=False,
            forward_version=None):
    """Per-seed final test accuracies as rows {model, seed, test_acc, run_dir}."""
    rows = []
    fv = forward_version or cfg.FORWARD_VERSION
    for mkey in cfg.MODEL_PREFIXES:
        run_dirs = cfg.resolve_run_dirs(mkey, exp_dir, seeds)
        if not run_dirs:
            print(f"[warn] no runs for {mkey} under {os.path.join(exp_dir, cfg.RUN_ROOT)}")
        for seed, rd in sorted(run_dirs.items()):
            acc, cached = cfg.get_test_accuracy(mkey, rd, ckpt, recompute=recompute,
                                                forward_version=fv)
            print(f"  {mkey:12s} seed {seed}: test={acc:.2f}%"
                  f"  ({'cached' if cached else 'evaluated'}, {os.path.basename(rd)})")
            rows.append({"model": mkey, "seed": seed, "test_acc": acc, "run_dir": rd})
    return pd.DataFrame(rows)


def plot_acc_bars(df, cfg, out_dir):
    """One bar per family: mean ± std over seeds, annotated with the value and n.
    Writes ``final_acc_bars.svg`` + ``final_acc_bars.csv``."""
    if df.empty:
        print("[warn] no runs found, nothing to plot")
        return

    spec = cfg.SPEC
    agg = (df.groupby("model")["test_acc"]
             .agg(mean="mean", std="std", n="count").reset_index())
    agg["std"] = agg["std"].fillna(0.0)   # a single seed has no spread
    agg["sem"] = agg["std"] / np.sqrt(agg["n"])
    # keep the canonical family order (axonal learned/fixed, synaptic learned/fixed)
    order = [m for m in cfg.MODEL_PREFIXES if m in set(agg["model"])]
    agg = agg.set_index("model").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=style.FIGSIZE_LINE)
    xs = np.arange(len(agg))
    bars = ax.bar(xs, agg["mean"], yerr=agg["std"], capsize=6,
                  color=[spec.color(m) for m in agg["model"]], alpha=0.85)
    for b, mkey in zip(bars, agg["model"]):
        h = spec.hatch(mkey)
        if h:
            b.set_hatch(h)
    for x, m, s in zip(xs, agg["mean"], agg["std"]):
        ax.text(x, m + s + 0.3, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{spec.label(m)}\n(n={n})" for m, n in zip(agg["model"], agg["n"])])
    ax.set_ylabel(f"{cfg.DATASET} test accuracy (%)")
    ax.set_title(f"{cfg.DATASET} final test accuracy (mean ± std over seeds)")
    ax.set_ylim(style.ACC_YLIM)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "final_acc_bars.svg"))
    agg.to_csv(os.path.join(out_dir, "final_acc_bars.csv"), index=False)
    print("\n" + agg.to_string(index=False))


def run_accbars_analysis(cfg, out_dir, exp_dir=".", seeds=None, ckpt="best.pth",
                         recompute=False, forward_version=None):
    """Collect the per-seed test accuracies and write the bar figure + CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    df = collect(cfg, exp_dir, seeds, ckpt, recompute, forward_version)
    df.to_csv(os.path.join(out_dir, "seed_summary.csv"), index=False)
    plot_acc_bars(df, cfg, out_dir)
    return df
