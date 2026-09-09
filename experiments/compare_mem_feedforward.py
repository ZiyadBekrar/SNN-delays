"""Feedforward-only memorization comparison; all measurements use training data.

Trains four feedforward networks on one fixed random-label dataset and plots them
together:

    No delays    SNN                          (plain LIF stack, no delays)
    Synaptic FF  SNN_feedforward_delays        (one learned delay per connection)
    Axonal FF    SNN_axonal_feedforward_delays (one learned delay per source neuron)
    Hybrid FF    SNN_feedforward_hybrid        (learned axonal delay + fixed random
                                                per-synapse offset, d_ij = d_j + delta_ij)

This is the feedforward analogue of experiments/compare_mem_delays.py (which
compares axonal/synaptic/hybrid delays on BOTH pathways). Here there is no
recurrence and no matched-initialisation step: the four architectures do not share
a layer structure, so each is built from the same RNG seed and trained on the same
data, exactly as a single experiments/train_mem.py run would.

Run: python experiments/compare_mem_feedforward.py
     python experiments/compare_mem_feedforward.py --epochs 100 --hidden-layers 64,64
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from train_mem import ROOT, Config, networks, run, torch, plt
from delrec.networks import dcls_module, learned_delay_parameter

MODELS = [
    ("No delays",   "SNN"),
    ("Synaptic FF", "SNN_feedforward_delays"),
    ("Axonal FF",   "SNN_axonal_feedforward_delays"),
    ("Hybrid FF",   "SNN_feedforward_hybrid"),
]
COLORS = ["tab:gray", "tab:orange", "tab:blue", "tab:green"]


def delay_param_count(model):
    """Number of optimised feedforward-delay parameters (0 for the plain SNN)."""
    return sum(learned_delay_parameter(m, "P").numel()
              for m in model.layers if isinstance(m, dcls_module))


def fixed_offset_count(model):
    """Number of frozen per-synapse offset entries (hybrid only)."""
    return sum(b.numel() for name, b in model.named_buffers() if name.endswith(".offsets"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("epochs", "seed", "dataset-seed", "num-samples",
                 "hybrid-max-synaptic-delay", "hybrid-delay-seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--task-type", choices=["temporal", "spatial"])
    parser.add_argument("--hidden-layers", help="Comma-separated widths, e.g. 64,64")
    parser.add_argument("--readout", choices=["mean", "sum", "last"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = Config()
    for key in ("epochs", "seed", "dataset_seed", "num_samples", "task_type", "readout",
                "hybrid_max_synaptic_delay", "hybrid_delay_seed"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    if args.hidden_layers:
        config.hidden_layers = [int(n) for n in args.hidden_layers.split(",")]
    if min(config.epochs, config.num_samples, *config.hidden_layers) < 1:
        parser.error("Epochs, samples and layer widths must be positive")
    torch.set_num_threads(config.cpu_threads)

    out = args.out or ROOT / "exp" / "MEM" / "feedforward_comparison" / (
        f"{config.task_type}_seed{config.seed}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    results, histories = {}, {}
    for label, model_name in MODELS:
        cfg = deepcopy(config)
        cfg.model = model_name
        history, final, run_dir = run(cfg, device, out / label.lower().replace(" ", "_"))
        model = getattr(networks, model_name)(cfg)  # a fresh instance just for the counts
        final["feedforward_delay_parameters"] = delay_param_count(model)
        final["fixed_synaptic_offsets"] = fixed_offset_count(model)
        results[label] = final
        histories[label] = history
        print(f"{label:>11}: acc={final['accuracy_percent']:.2f}%  "
              f"trainable={final['trainable_parameters']:,}  "
              f"delay_params={final['feedforward_delay_parameters']:,}  "
              f"fixed_offsets={final['fixed_synaptic_offsets']:,}", flush=True)

    # Every run must have trained on byte-identical data.
    reference = torch.load(out / MODELS[0][0].lower().replace(" ", "_") / "dataset.pt",
                           weights_only=True)
    for label, _ in MODELS[1:]:
        other = torch.load(out / label.lower().replace(" ", "_") / "dataset.pt",
                           weights_only=True)
        assert all(torch.equal(reference[k], other[k]) for k in reference), \
            f"dataset mismatch for {label}"

    (out / "comparison.json").write_text(json.dumps({
        "note": "Feedforward-only models trained on one fixed random-label dataset. "
                "No matched initialisation: each model is built from config.seed and "
                "trained on the same data, as in a single train_mem.py run.",
        "models": {label: name for label, name in MODELS},
        "results": results,
    }, indent=2))

    plot_comparison(histories, results, config, out)


def plot_comparison(histories, results, config, out):
    """Render current or saved comparison metrics without rerunning training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for (label, history), color in zip(histories.items(), COLORS):
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["loss"] for r in history], label=label, color=color)
        axes[1].plot(epochs, [r["accuracy_percent"] for r in history], label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Training accuracy", ylim=(0, 105))
    axes[1].axhline(100 / config.output_size, color="gray", linestyle="--",
                    label="Uniform random guess")
    for axis in axes[:2]:
        axis.legend()
        axis.grid(alpha=0.25)

    values = [r["accuracy_percent"] for r in results.values()]
    labels = [f"{name}\n{result['trainable_parameters']:,} params"
              for name, result in results.items()]
    bars = axes[2].bar(labels, values, color=COLORS)
    axes[2].bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4)
    axes[2].set(ylabel="Accuracy (%)", title="Final training accuracy", ylim=(0, 110))

    fig.suptitle(f"Feedforward delays on the memorization task | {config.task_type}, "
                 f"{config.num_samples} samples, topology {config.input_size} → "
                 + " → ".join(map(str, config.hidden_layers + [config.output_size])))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"feedforward_comparison.{extension}", dpi=180)
    plt.close(fig)
    print(f"Comparison saved: {out}", flush=True)


if __name__ == "__main__":
    main()
