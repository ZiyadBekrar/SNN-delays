"""Time and memory of the recurrent-delay kernels on real dataset batches.

Companion to ``profile_kernels.py``, which sweeps a single isolated layer on synthetic
current. This one runs the forward paths inside the full SSC, HAR and PS-MNIST models,
at each dataset's own operating point, with the real loss, and measures per-batch
wall-clock time and transient GPU memory for forward and forward+backward. The
delay-free ``vanilla_recurrent`` stack is the baseline.

Regime: ``use_sig_p=False``, ``decay_input=False``, no dropout, no recurrent bias, so
every mode is numerically identical and directly comparable.

Outputs:  exp/profiling/<timestamp>/                          [GPU + datasets]
Usage:    CUDA_VISIBLE_DEVICES=0 python experiments/profiling/profile_kernels_datasets.py
"""

import os
from common import paths
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Central graphic chart (experiments/make_figures/common/style.py applies its rcParams on import).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]
from common import style

import delrec.networks as snn_module
from delrec.delay_layers import axonal_recdel, vanilla_recurrent
from delrec.datasets import load_dataset
from delrec.utils import reset_states, seed_everything, calc_loss_SSC, calc_loss_HAR

# Reuse the exact mode table (label/color/marker/linestyle) from the synthetic
# profiler so the two benchmarks read as one system.
from profile_kernels import MODES_BY_KIND, MIB, gpu_tag


# --------------------------------------------------------------------------- #
# Benchmark configuration
# --------------------------------------------------------------------------- #
# (dataset tag, config import path, real-loss fn). SNN_recurrent_delays (axonal)
# and SNN_synaptic_recurrent_delays (synaptic) are the same classes HAR/SSC train.
DATASETS = [
    ("SSC", "configs.perf_SSC", calc_loss_SSC),
    ("HAR", "configs.perf_HAR", calc_loss_HAR),
    ("PSMNIST", "configs.perf_PSMNIST", calc_loss_SSC),   # reuses SSC's snn.py + loss
]
KINDS = [
    ("axonal",   "SNN_recurrent_delays"),
    ("synaptic", "SNN_synaptic_recurrent_delays"),
]

N_BATCHES = int(os.environ.get("REAL_N_BATCHES", "4"))   # distinct real batches timed
WARMUP = 2              # untimed iters (compiles Triton kernels, warms caches)
REPEATS = 3             # timed iters per batch. Per-batch time = median over all
SEED = 0


# --------------------------------------------------------------------------- #
# Model / data construction (fair-comparison regime)
# --------------------------------------------------------------------------- #
def build_config(config_path):
    """Import the dataset's Config and force the fair-comparison kernel regime."""
    mod = __import__(config_path, fromlist=["Config"])
    cfg = mod.Config()
    cfg.seed = SEED
    # All four axonal modes are numerically identical only in this regime.
    cfg.use_sig_p = False
    cfg.decay_input = False
    cfg.recurrent_dropout_rate = 0.0
    cfg.feedforward_dropout_rate = 0.0    # isolate the kernel, common to all modes
    cfg.use_rec_bias = False
    return cfg


def build_model(cfg, model_name, device):
    seed_everything(seed=SEED, is_cuda=True)
    model = getattr(snn_module, model_name)(cfg).to(device)
    model.train()          # build the backward graph in fwd+bwd runs
    return model


def build_vanilla_model(cfg, device):
    """Full model with every recurrent-delay layer replaced by a plain no-delay
    `vanilla_recurrent` layer (pure-PyTorch LIF loop). The Linear/BN/readout stack
    is untouched, so this isolates the cost of the whole delay machinery. Same for
    both kinds (vanilla has no delay type), so it's the common baseline floor."""
    seed_everything(seed=SEED, is_cuda=True)
    model = snn_module.SNN_recurrent_delays(cfg).to(device)
    new_layers = [vanilla_recurrent(cfg, m.neurons, cfg.neuron_module).to(device)
                  if isinstance(m, axonal_recdel) else m
                  for m in model.layers]
    model.layers = torch.nn.Sequential(*new_layers)
    model.train()
    return model


def get_batches(ds, cfg, device):
    """A fixed list of real (inputs (T,B,N), targets) batches from the train split.
    Fetched once so data loading isn't timed and every mode sees identical inputs."""
    seed_everything(seed=SEED, is_cuda=True)
    train_loader, _, _ = load_dataset(cfg)
    batches = []
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= N_BATCHES:
            break
        if ds == "PSMNIST":
            # MNIST yields (B, 1, 28, 28). Flatten to the pixel sequence (B, T, 1)
            # as src/PSMNIST/trainer.py does. The per-run pixel permutation only
            # reindexes the time axis, so it's irrelevant to timing/memory, skip it.
            inputs = inputs.view(-1, cfg.time_window, cfg.input_size)
        inputs = inputs.permute(1, 0, 2).float().to(device)   # (T, B, N)
        targets = targets.to(device)
        batches.append((inputs, targets))
    return batches


def set_mode(model, mode):
    """Point every recurrent-delay layer (axonal + synaptic subclass) at `mode`."""
    for m in model.modules():
        if isinstance(m, axonal_recdel):
            m.forward_version = mode


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _run_once(model, x, y, loss_fn, backward):
    """One reset + real forward (+ real loss & backward)."""
    reset_states(model)
    if backward:
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss_fn(out, y).backward()
    else:
        with torch.no_grad():
            out = model(x)
    return out


def _firing_rate(model, x, y, loss_fn):
    """Mean output spike rate across the recurrent-delay layers on one batch."""
    acc = []

    def hook(_m, _inp, out):
        acc.append(float(out.detach().float().mean().item()))

    handles = [m.register_forward_hook(hook)
               for m in model.modules()
               if isinstance(m, (axonal_recdel, vanilla_recurrent))]
    try:
        _run_once(model, x, y, loss_fn, backward=False)
    finally:
        for h in handles:
            h.remove()
    return float(np.mean(acc)) if acc else float("nan")


def bench(model, batches, mode, loss_fn, backward, device):
    """Median per-batch time (ms) and peak transient memory (MiB) for one mode."""
    set_mode(model, mode)
    try:
        # Warmup on the first batch (compiles Triton kernels, warms caches/cuBLAS).
        x0, y0 = batches[0]
        for _ in range(WARMUP):
            _run_once(model, x0, y0, loss_fn, backward)
        torch.cuda.synchronize()

        # Timing: every real batch x REPEATS, median over all (CUDA events).
        times = []
        for x, y in batches:
            for _ in range(REPEATS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _run_once(model, x, y, loss_fn, backward)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))   # ms
        times.sort()
        t_ms = times[len(times) // 2]

        # Peak transient (activation) memory = peak-during-op minus resident-before.
        x, y = batches[0]
        reset_states(model)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        _run_once(model, x, y, loss_fn, backward)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        mem_mib = (peak - before) / MIB

        fr = _firing_rate(model, x, y, loss_fn)
        return t_ms, mem_mib, fr
    except Exception as e:
        # CUDA OOM or a Triton shared-memory/resource limit -> record N/A so the
        # plot shows the kernel dropping out (matches profile_kernels.py).
        msg = str(e).lower()
        if "out of memory" in msg or "out of resource" in msg:
            torch.cuda.empty_cache()
            return float("nan"), float("nan"), float("nan")
        raise


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_all(device):
    rows = []
    for ds, config_path, loss_fn in DATASETS:
        cfg = build_config(config_path)
        batches = get_batches(ds, cfg, device)
        T, B, N = batches[0][0].shape
        print(f"\n########## {ds}  (T={T}, B={B}, input={N}, "
              f"hidden={cfg.hidden_layers}) ##########")
        for kind, model_name in KINDS:
            modes = MODES_BY_KIND[kind]
            model = build_model(cfg, model_name, device)
            vanilla_model = (build_vanilla_model(cfg, device)
                             if any(m[0] == "vanilla" for m in modes) else None)
            n_recdel = sum(isinstance(m, axonal_recdel) for m in model.modules())
            print(f"\n=== {ds} | {kind} ({model_name}) | "
                  f"{n_recdel} recdel layers | {len(modes)} modes ===")
            for backward in (False, True):
                regime = "fwdbwd" if backward else "fwd"
                for key, label, color, marker, ls in modes:
                    mdl = vanilla_model if key == "vanilla" else model
                    t_ms, mem_mib, fr = bench(mdl, batches, key, loss_fn,
                                              backward, device)
                    rows.append(dict(dataset=ds, kind=kind, mode=key, regime=regime,
                                     T=T, B=B, time_ms=t_ms, mem_mib=mem_mib,
                                     firing=fr))
                    print(f"  {regime:<7} {key:<13} {t_ms:9.3f} ms  "
                          f"{mem_mib:10.1f} MiB  (fr={fr:.3f})")
            del model
            if vanilla_model is not None:
                del vanilla_model
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _bar_panel(ax, sub, regime, metric, ylabel, modes, logy):
    d = sub[sub["regime"] == regime].set_index("mode")
    labels, colors, vals = [], [], []
    for key, label, color, marker, ls in modes:
        if key in d.index:
            labels.append(label)
            colors.append(color)
            vals.append(float(d.loc[key, metric]))
    xpos = range(len(labels))
    bars = ax.bar(xpos, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="major", alpha=0.5)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7,
                        xytext=(0, 1), textcoords="offset points")


def plot_bench(sub, dataset, kind, modes, out_dir, fr, T, B):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    _bar_panel(axes[0, 0], sub, "fwd", "time_ms", "per-batch time (ms)", modes, True)
    _bar_panel(axes[0, 1], sub, "fwdbwd", "time_ms", "per-batch time (ms)", modes, True)
    _bar_panel(axes[1, 0], sub, "fwd", "mem_mib", "peak memory (MiB)", modes, False)
    _bar_panel(axes[1, 1], sub, "fwdbwd", "mem_mib", "peak memory (MiB)", modes, False)

    axes[0, 0].set_title("forward", fontsize=11)
    axes[0, 1].set_title("forward + backward", fontsize=11)

    fig.suptitle(f"{dataset} — {kind} recurrent-delay kernels (real batches)",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.005, f"{dataset} operating point: T={T}, B={B}, seed={SEED}   ·   "
             f"{kind}, use_sig_p=False, ~{100*fr:.0f}% recurrent firing, "
             f"median of {N_BATCHES}×{REPEATS} real batches",
             ha="center", fontsize=8, color=style.INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"bench_{dataset}_{kind}.{ext}")
    plt.close(fig)


def main():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")
    print(f"Datasets: {[d[0] for d in DATASETS]} | kinds: {[k[0] for k in KINDS]} | "
          f"real batches: {N_BATCHES} × {REPEATS} repeats | seed={SEED}")

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    datasets = "-".join(d[0] for d in DATASETS)
    run_dir = Path(paths.runs("profiling")) / f"datasets_{datasets}_{gpu_tag()}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {run_dir}")

    df = run_all(device)
    df.to_csv(run_dir / "bench_datasets.csv", index=False)

    for ds, _cfg_path, _loss in DATASETS:
        for kind, _model_name in KINDS:
            sub = df[(df["dataset"] == ds) & (df["kind"] == kind)]
            if not len(sub):
                continue
            out_dir = run_dir / ds / kind
            out_dir.mkdir(parents=True, exist_ok=True)
            fr = float(sub["firing"].dropna().median()) if sub["firing"].notna().any() else 0.0
            T, B = int(sub["T"].iloc[0]), int(sub["B"].iloc[0])
            plot_bench(sub, ds, kind, MODES_BY_KIND[kind], out_dir, fr, T, B)

    print(f"\nAll done. Results under {run_dir} "
          f"(bench_datasets.csv + <dataset>/<kind>/ figures).")


if __name__ == "__main__":
    main()
