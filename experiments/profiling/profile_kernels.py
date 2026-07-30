"""Time and memory of one recurrent layer, sweeping one factor at a time.

Measures a full forward and backward for each implementation against a delay-free
recurrent layer as the floor, sweeping sequence length, batch size, layer width,
maximum delay and delay spread around a reference configuration.

Measurements are hardware-specific. The numbers recorded with this repository come
from one machine.

Outputs:  trained_models/profiling/<timestamp>/
Usage:    MPLBACKEND=Agg python experiments/profiling/profile_kernels.py
"""

import os
from common import paths
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Central graphic chart (experiments/make_figures/common/style.py applies its rcParams on import).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]
from common import style

from configs.perf_SSC import Config
from delrec.delay_layers import axonal_recdel, synaptic_recdel, vanilla_recurrent
from delrec.utils import reset_states, seed_everything

# --------------------------------------------------------------------------- #
# Benchmark configuration
# --------------------------------------------------------------------------- #
# SSC-sized operating point. Each sweep varies ONE of these, holding the rest.
DEFAULTS = dict(T=250, B=256, N=256, D=25, sigma=1.0)

SWEEPS = {
    "T":     [50, 100, 250, 500, 1000],
    "B":     [32, 64, 128, 256, 512],
    "N":     [64, 128, 256, 512, 1024],
    "D":     [5, 10, 25, 50, 100],
    "sigma": [0.0, 1.0, 2.0, 4.0, 8.0],
}
XLABEL = {
    "T": "Sequence length  T",
    "B": "Batch size  B",
    "N": "Layer width  N",
    "D": "Max recurrent delay  D  (timesteps)",
    "sigma": "Delay spread  σ",
}
TITLE = {
    "T": "Dependence on sequence length T",
    "B": "Dependence on batch size B",
    "N": "Dependence on layer width N",
    "D": "Dependence on delay range D",
    "sigma": "Dependence on delay spread σ",
}
LOGX = {"T": True, "B": True, "N": True, "D": True, "sigma": False}

INPUT_SCALE = 0.8       # ~12% recurrent firing rate at the default operating point
WARMUP = 3              # untimed iters (compiles Triton kernels, warms caches/cuBLAS)
REPEATS = 10            # timed iters. Per-batch time = median
SEED = 0

# mode key -> (legend label, color, marker, linestyle). Colors drawn from the
# repo chart's two base hues plus its two extra categoricals. The reference paths
# (v1/v2) are the cool family, the fast kernels the warm/green family, and the
# delay-free `vanilla_recurrent` baseline is gray/dotted (the delay-free-control style).
_V2      = ("v2",           "v2  (torch ref)",     style.BLUE,      "o", "-")
_V1      = ("v1",           "v1  (buffer ref)",    "#4a3aa7",       "s", "--")
_EXACT   = ("triton_exact", "triton exact",        style.ORANGE,    "^", "-")
_ED      = ("eventdriven",  "event-driven",        "#1baf7a",       "D", "-")
_VANILLA = ("vanilla",      "vanilla (no delay)",  style.INK_MUTED, "P", ":")

# Per layer kind. Synaptic_recdel.forward has NO triton_exact branch, a fused
# per-synapse Triton scan loses to v2's per-step GEMM, so it silently falls
# through to v2 (see recurrent_neurons.py). Profiling it would just duplicate the
# v2 curve mislabeled "triton exact", so synaptic gets three kernel modes, axonal
# four. `vanilla` is the same pure-PyTorch no-delay layer in both, a common floor.
MODES_BY_KIND = {
    "axonal":   [_V2, _V1, _EXACT, _ED, _VANILLA],
    "synaptic": [_V2, _V1, _ED, _VANILLA],
}

# (subfolder tag, layer class), each profiled independently into its own folder.
LAYER_KINDS = [
    ("axonal",   axonal_recdel),
    ("synaptic", synaptic_recdel),
]

MIB = 1024 ** 2


# --------------------------------------------------------------------------- #
# Layer / input construction
# --------------------------------------------------------------------------- #
def base_config():
    cfg = Config()
    # Regime shared by all four kernels (axonal event-driven requires it).
    cfg.use_sig_p = False
    cfg.decay_input = False
    cfg.recurrent_dropout_rate = 0.0     # isolate the kernel, identical across modes
    cfg.use_rec_bias = False
    # Delay range is controlled explicitly via a uniform init on [0, D].
    cfg.init_rec_delay = "uniform"
    cfg.init_recdel_offset = 0
    return cfg


def build_layer(layer_cls, N, D, sigma, device):
    cfg = base_config()
    cfg.max_rec_delay = D
    cfg.sigma_init = float(sigma)
    seed_everything(seed=SEED, is_cuda=True)
    layer = layer_cls(cfg, N).to(device)
    layer.sigma = float(sigma)   # explicit (use_sig_p=False -> sigma is scalar width)
    layer.train()                # build the backward graph in fwd+bwd runs
    return layer


def build_vanilla(N, device):
    """Plain no-delay recurrent SNN layer (pure-PyTorch LIF loop), the baseline
    floor. Same regime/config as the delay layers. Only the delays are absent."""
    cfg = base_config()
    seed_everything(seed=SEED, is_cuda=True)
    layer = vanilla_recurrent(cfg, N).to(device)
    layer.train()                # build the backward graph in fwd+bwd runs
    return layer


def make_input(T, B, N, device):
    seed_everything(seed=SEED + 1, is_cuda=True)
    return torch.randn(T, B, N, device=device) * INPUT_SCALE


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _run_once(layer, x, backward):
    """One reset + forward (+ backward). Returns the output for firing-rate read."""
    reset_states(layer)
    if backward:
        layer.zero_grad(set_to_none=True)
        y = layer(x)
        y.sum().backward()
    else:
        with torch.no_grad():
            y = layer(x)
    return y


def bench(layer, x, mode, backward, device):
    """Median per-batch time (ms) and peak transient memory (MiB) for one mode."""
    if mode != "vanilla":        # vanilla_recurrent has a single pure-torch path
        layer.forward_version = mode
    try:
        for _ in range(WARMUP):
            _run_once(layer, x, backward)
        torch.cuda.synchronize()

        # Timing (CUDA events, median of REPEATS).
        times = []
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_once(layer, x, backward)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))   # ms
        times.sort()
        t_ms = times[len(times) // 2]

        # Peak transient (activation) memory = peak-during-op minus resident-before.
        reset_states(layer)
        layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        y = _run_once(layer, x, backward)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        mem_mib = (peak - before) / MIB
        fr = float(y.detach().mean().item())
        del y
        return t_ms, mem_mib, fr
    except Exception as e:
        # CUDA OOM or a Triton shared-memory/resource limit (triton_exact packs
        # the whole delay window into shared memory, so large L is infeasible on
        # this GPU). Record N/A and let the plot show the kernel dropping out.
        msg = str(e).lower()
        if "out of memory" in msg or "out of resource" in msg:
            torch.cuda.empty_cache()
            return float("nan"), float("nan"), float("nan")
        raise


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def run_all(device, kind, layer_cls, modes):
    """Run every sweep for ONE layer kind. Returns a DataFrame tagged with `kind`."""
    mode_keys = [m[0] for m in modes]
    rows = []
    for var, values in SWEEPS.items():
        print(f"\n=== [{kind}] sweep {var} : {values} ===")
        for val in values:
            params = dict(DEFAULTS)
            params[var] = val
            T, B, N, D, sigma = (params[k] for k in ("T", "B", "N", "D", "sigma"))

            layer = build_layer(layer_cls, N, D, sigma, device)
            x = make_input(T, B, N, device)
            L = int(torch.ceil(1.0 + layer.recurrent_delays.max() + (1.0 + layer.sigma)).item()) + 1
            # The no-delay baseline is a distinct layer. It ignores D and sigma, so
            # its curve is flat along those sweeps (a meaningful reference floor).
            vlayer = build_vanilla(N, device) if "vanilla" in mode_keys else None

            for backward in (False, True):
                regime = "fwdbwd" if backward else "fwd"
                for mode in mode_keys:
                    lyr = vlayer if mode == "vanilla" else layer
                    t_ms, mem_mib, fr = bench(lyr, x, mode, backward, device)
                    rows.append(dict(kind=kind, sweep=var, value=val, T=T, B=B, N=N,
                                     D=D, sigma=sigma, L=L, regime=regime, mode=mode,
                                     time_ms=t_ms, mem_mib=mem_mib, firing=fr))
                    print(f"  {var}={val!s:<5} {regime:<7} {mode:<13} "
                          f"{t_ms:8.3f} ms  {mem_mib:9.1f} MiB  (fr={fr:.3f})")
            del layer, x
            if vlayer is not None:
                del vlayer
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _panel(ax, df, var, regime, metric, ylabel, logy, modes):
    sub = df[(df["sweep"] == var) & (df["regime"] == regime)]
    for key, label, color, marker, ls in modes:
        d = sub[sub["mode"] == key].sort_values("value")
        ax.plot(d["value"], d[metric], marker=marker, ls=ls, color=color,
                label=label, lw=1.8, ms=5.5, markeredgecolor="white",
                markeredgewidth=0.5)
    if LOGX[var]:
        ax.set_xscale("log")
        ax.set_xticks(SWEEPS[var])
        ax.set_xticklabels([str(v) for v in SWEEPS[var]])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.5)
    ax.margins(x=0.05)


def plot_sweep(df, var, out_dir, modes, kind, fr):
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex="col")
    # Row 0: per-batch time (log y). Row 1: peak transient memory (linear y).
    _panel(axes[0, 0], df, var, "fwd", "time_ms", "per-batch time (ms)", True, modes)
    _panel(axes[0, 1], df, var, "fwdbwd", "time_ms", "per-batch time (ms)", True, modes)
    _panel(axes[1, 0], df, var, "fwd", "mem_mib", "peak memory (MiB)", False, modes)
    _panel(axes[1, 1], df, var, "fwdbwd", "mem_mib", "peak memory (MiB)", False, modes)

    axes[0, 0].set_title("forward", fontsize=11)
    axes[0, 1].set_title("forward + backward", fontsize=11)
    for ax in axes[1, :]:
        ax.set_xlabel(XLABEL[var])

    held = ", ".join(f"{k}={DEFAULTS[k]}" for k in ("T", "B", "N", "D", "sigma")
                     if k != var)
    fig.suptitle(f"{kind} recurrent-delay kernels — {TITLE[var]}", fontsize=13, y=0.99)
    fig.text(0.5, 0.005, f"held fixed: {held}, seed={SEED}   ·   {kind}, "
             f"use_sig_p=False, ~{100*fr:.0f}% firing, "
             f"median of {REPEATS}",
             ha="center", fontsize=8, color=style.INK_MUTED)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(modes), frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"kernels_sweep_{var}.{ext}")
    plt.close(fig)


# Default firing rate for the caption, replaced by the measured value per kind.
INPUT_SCALE_FR = 0.12


def gpu_tag():
    """Filesystem-safe short GPU name for run-dir labels, e.g. 'GB10', 'A100-SXM4'."""
    import re
    name = torch.cuda.get_device_name(0).replace("NVIDIA", "").strip()
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "gpu"


def _default_firing(df):
    """Measured firing rate at the default operating point (T-sweep midpoint, v2)."""
    dflt = df[(df["sweep"] == "T") & (df["value"] == DEFAULTS["T"]) &
              (df["regime"] == "fwd") & (df["mode"] == "v2")]
    return float(dflt["firing"].iloc[0]) if len(dflt) else INPUT_SCALE_FR


def main():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")
    print("Default operating point: "
          + ", ".join(f"{k}={v}" for k, v in DEFAULTS.items())
          + f", seed={SEED}")
    print("Sweeps (one axis varied at a time): "
          + ", ".join(f"{k}={v}" for k, v in SWEEPS.items()))

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = Path(paths.runs("profiling")) / f"sweeps_{gpu_tag()}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {run_dir}")

    # One self-contained subfolder per layer kind (axonal / synaptic).
    for kind, layer_cls in LAYER_KINDS:
        modes = MODES_BY_KIND[kind]
        print(f"\n########## {kind.upper()} ({len(modes)} modes) ##########")
        out_dir = run_dir / kind
        out_dir.mkdir(parents=True, exist_ok=True)

        df = run_all(device, kind, layer_cls, modes)
        df.to_csv(out_dir / "kernel_profile.csv", index=False)

        fr = _default_firing(df)
        for var in SWEEPS:
            plot_sweep(df, var, out_dir, modes, kind, fr)
        print(f"\n[{kind}] done. {len(SWEEPS)} figures + kernel_profile.csv in {out_dir}")

    print(f"\nAll done. Results under {run_dir} (axonal/ and synaptic/).")


if __name__ == "__main__":
    main()
