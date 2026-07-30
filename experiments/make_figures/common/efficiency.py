"""Spiking-efficiency and per-class analyses for DelRec models.

Two views that both need one forward pass over the test set (no perturbation):

  * Efficiency. The hidden-layer firing rate (mean spikes per neuron per
    timestep, read from the ``spike_registrator`` modules) plotted against test
    accuracy. If learned delays reach the same accuracy at a lower firing rate,
    that is an energy advantage independent of the (small) accuracy gap.
  * Per-class accuracy, Δaccuracy (learned − fixed) per class, localizing the
    advantage to the temporally-structured activities rather than resting on the
    mean.

The two facades ``run_efficiency_analysis`` / ``run_per_class_analysis`` take tidy
DataFrames (one row per run / per run×class) so the HAR driver only has to collect
them.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style


# --------------------------------------------------------------------------- #
# Single-forward evaluation: firing rate + per-class correct counts
# --------------------------------------------------------------------------- #
def evaluate_efficiency(loader, model, device, n_classes):
    """One forward pass over the test set."""
    import torch
    from delrec.networks import spike_registrator
    from delrec.utils import reset_states

    registrators = [m for m in model.modules() if isinstance(m, spike_registrator)]
    model.eval()
    if hasattr(loader, "reset"):
        loader.reset()

    spikes_sum = 0.0
    spikes_elems = 0
    cls_correct = np.zeros(n_classes)
    cls_total = np.zeros(n_classes)
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.permute(1, 0, 2).float().to(device)
            targets = targets.to(device)
            reset_states(model=model)
            outputs = model(inputs)                          # (T, B, C)
            pred = outputs.mean(0).argmax(1)                 # (B,)
            hit = pred.eq(targets)
            correct += hit.sum().item()
            total += targets.size(0)
            for c in targets.unique():
                m = targets == c
                cls_correct[c.item()] += hit[m].sum().item()
                cls_total[c.item()] += m.sum().item()
            for reg in registrators:
                s = reg.spikes
                spikes_sum += s.sum().item()
                spikes_elems += s.numel()

    firing_rate = spikes_sum / max(spikes_elems, 1)
    per_class = 100.0 * cls_correct / np.where(cls_total > 0, cls_total, 1)
    per_class[cls_total == 0] = np.nan
    return 100.0 * correct / total, firing_rate, per_class


# --------------------------------------------------------------------------- #
# Efficiency plots
# --------------------------------------------------------------------------- #
def plot_accuracy_vs_firing(runs, out_dir, spec):
    """Scatter of test accuracy vs firing rate, one point per run. Color = delay
    type, filled = learned / open = fixed (the condition channel)."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    seen = set()
    for _, r in runs.iterrows():
        mkey = r["model"]
        c = spec.color(mkey)
        learned = style.condition_of(mkey) == "learned"
        ax.scatter(r["firing_rate"], r["acc"], s=42, color=c if learned else "none",
                   edgecolor=c, linewidth=1.4, marker="o" if learned else "s",
                   alpha=0.9, label=spec.label(mkey) if mkey not in seen else None)
        seen.add(mkey)
    ax.set_xlabel("Hidden firing rate (spikes / neuron / step)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs spiking activity (one point per run)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "accuracy_vs_firing.svg"))


def plot_firing_bar(runs, out_dir, spec):
    """Mean firing rate per family (mean ± std over runs)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, means, stds, labels, colors, hatches = [], [], [], [], [], []
    for i, mkey in enumerate(runs["model"].unique()):
        fr = runs[runs["model"] == mkey]["firing_rate"].to_numpy()
        xs.append(i)
        means.append(fr.mean())
        stds.append(fr.std())
        labels.append(f"{spec.label(mkey)}\n(n={len(fr)})")
        colors.append(spec.color(mkey))
        hatches.append(spec.hatch(mkey))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hidden firing rate (spikes / neuron / step)")
    ax.set_title("Spiking activity per family (mean ± std over runs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "firing_bar.svg"))


def run_efficiency_analysis(runs, out_dir, spec):
    """Write the efficiency figures + CSV from a runs DataFrame (columns
    model/std/seed/acc/firing_rate)."""
    os.makedirs(out_dir, exist_ok=True)
    runs.to_csv(os.path.join(out_dir, "efficiency_per_run.csv"), index=False)
    plot_accuracy_vs_firing(runs, out_dir, spec)
    plot_firing_bar(runs, out_dir, spec)


# --------------------------------------------------------------------------- #
# Per-class accuracy
# --------------------------------------------------------------------------- #
def run_per_class_analysis(per_class, out_dir, spec, modalities):
    """Δaccuracy (learned − fixed) per class, one bar chart per modality, from a
    per_class DataFrame (columns model/std/seed/class/acc). Δ is averaged over stds
    and seeds. The ±std is over that pool."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    per_class.to_csv(os.path.join(out_dir, "per_class_per_run.csv"), index=False)

    # mean per-class accuracy per model (pooled over std × seed).
    mean_acc = per_class.groupby(["model", "class"])["acc"].agg(["mean", "std"]).reset_index()

    for modality, (fixed_key, learned_key) in modalities.items():
        f = mean_acc[mean_acc["model"] == fixed_key].set_index("class")
        l = mean_acc[mean_acc["model"] == learned_key].set_index("class")
        classes = sorted(set(f.index).intersection(l.index))
        if not classes:
            continue
        delta = np.array([l.loc[c, "mean"] - f.loc[c, "mean"] for c in classes])
        err = np.array([np.hypot(l.loc[c, "std"], f.loc[c, "std"]) for c in classes])

        fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(classes) + 2), 4))
        colors = [style.DIVERGING(0.85 if d >= 0 else 0.15) for d in delta]
        ax.bar(range(len(classes)), delta, yerr=err, capsize=3, color=colors, alpha=0.9)
        ax.axhline(0, color=style.INK_MUTED, lw=0.8)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([str(c) for c in classes], fontsize=8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Δ accuracy: learned − fixed (pts)")
        ax.set_title(f"{modality}: per-class accuracy gain from learned delays")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"per_class_delta_{modality}.svg"))
