"""Sweep over s_init x discretization scheme.

Produces: straight-through and stochastic rounding, against nearest.
Outputs:  trained_models/HAR/rounding_alternatives_sweep/
Usage:    python experiments/sweeps/sweep_HAR_rounding_alternatives.py

Each run writes the same artifacts as a single training run. The
`parallel_` twin of this script runs one process per run across the
available GPUs, giving each its own Triton cache so concurrent JITs do
not race.
"""

import os
import sys
# Repo-root on sys.path so `configs`/`delrec` resolve when run as `python experiments/sweeps/<script>.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]

import torch
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import numpy as np

from configs.perf_HAR import Config

from delrec.training.har import *
from delrec.networks import *
import delrec.networks as snn_module
from delrec.datasets import load_dataset


# Rounding-alternatives counterpart of sweep_HAR_delaystd.py, for the two LEARNED
# families only (axonal SNN_recurrent_delays, synaptic SNN_synaptic_recurrent_delays).
# It trains the same delay_std_init sweep but adds a THIRD axis: how the learnable
# recurrent delays are discretized onto the integer grid. Each cell is one
# (model, mode, delay_std_init, seed) run, landing in
#   exp/HAR/rounding_alternatives_sweep/<ModelClass>/<mode>_std<X>/seed<N>/
# so the two existing sweeps stay untouched as baselines:
#   exp/HAR/std_init_sweep/          nearest round every epoch (the 'nearest' mode here)
#   exp/HAR/std_init_sweep_noround/  no rounding at all, delays stay fractional
#
# The modes (see MODES below):
#   'ste', straight-through estimator: the recurrent forward taps at round(d)
#                  at EVERY step (train and eval alike), while the gradient flows
#                  straight through to the fractional recurrent_delays. The parameter
#                  is never rounded in place (round_pos_each_epoch=False), so the
#                  fractional master copy accumulates sub-step gradient pressure.
#                  Train and eval regimes are identical -> no discretization gap.
#   'stochastic', the existing per-epoch in-place projection, but stochastic:
#                  floor(d + U[0,1)), i.e. Round up with probability frac(d).
#                  Unbiased (E[SR(d)] = d), so a within-epoch drift of e.g. 0.3 moves
#                  the delay by 1 with probability 0.3 instead of being erased by
#                  nearest-rounding's deadzone. Measured per-epoch mean |Delta d| in
#                  the trained std_init_sweep is ~0.25-0.45, squarely in that regime.
#   'nearest', deterministic d.round(). The std_init_sweep baseline, kept here so
#                  the control can be re-run under identical code if wanted (it is NOT
#                  in the default mode list).
#
# Per run it persists, exactly like sweep_HAR_delaystd.py:
#   config.json, train_res.csv, val_res.csv, best.pth, last.pth
#   init_delays.npz, per-layer recurrent_delays right after init (the "beginning")
#   delay_variation.csv, per-epoch mean-abs delay variation per recdel layer
#   final_test.json, final test acc/loss + best val/epoch + the mode knobs
# A manifest_rounding_alternatives.json at the sweep root indexes all runs.
#
# ``run_single`` trains ONE (model, mode, delay_std_init, seed) run and returns its
# manifest row. It is reused by the sequential loop below and by
# parallel_sweep_HAR_rounding_alternatives.py (one process per run).
FORWARD_VERSION = os.environ.get("HAR_FORWARD_VERSION", "triton_exact")  # axonal: 'triton_exact'|'triton_approx'|'triton_ste'|'eventdriven'
# Per-synapse delays: the accelerated-exact path is 'eventdriven' (spike-sparse
# scatter, fastest). Applied to synaptic_recdel layers only.
SYN_FORWARD_VERSION = os.environ.get("HAR_SYN_FORWARD_VERSION", "eventdriven")  # 'eventdriven'|'v2'

# mode tag -> the three config knobs it sets. Round_mode is only read by round_pos(),
# so it is irrelevant when round_pos is False (the 'ste' row).
MODES = {
    'ste':        dict(round_pos=False, round_mode='nearest',    round_delays=True),
    'stochastic': dict(round_pos=True,  round_mode='stochastic', round_delays=False),
    'nearest':    dict(round_pos=True,  round_mode='nearest',    round_delays=False),
}


def mode_tag(mode, dsi):
    """Subdir tag for a (mode, delay_std_init) cell: ste_std8, stochastic_std13, ..."""
    return f"{mode}_std{dsi:g}"


def _recdel_layers(model):
    """Return the axonal_recdel layers in model.layers order (covers synaptic_recdel)."""
    return [m for m in model.layers if isinstance(m, axonal_recdel)]


def run_single(model_name, dsi, seed, sweep_root, device, *,
               mode='ste', epochs_override="", batch_override="", subdir=None):
    """Train a single (model, mode, delay_std_init, seed) run and persist all artifacts under
    ``sweep_root/model_name/<subdir>/seed<seed>/``.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown rounding mode '{mode}' (expected one of {sorted(MODES)}).")
    knobs = MODES[mode]
    tag = subdir if subdir is not None else mode_tag(mode, dsi)

    config = Config()
    config.seed = seed
    config.delay_std_init = dsi
    # --- the rounding regime under test ---
    config.round_pos_each_epoch = knobs['round_pos']
    config.round_mode = knobs['round_mode']
    config.round_delays = knobs['round_delays']
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    # NOTE: config.round_delays must be set BEFORE the model is built, axonal_recdel
    # reads it in __init__.
    model = getattr(snn_module, model_name)(config).to(device)

    # --- switch the recurrent-delay layers to their fast forward path ---
    # synaptic_recdel (per-synapse, subclass of axonal_recdel) -> eventdriven.
    # Axonal layers -> the Triton kernel.
    n_ax = n_syn = 0
    for m in model.modules():
        if isinstance(m, synaptic_recdel):
            m.forward_version = SYN_FORWARD_VERSION
            n_syn += 1
        elif isinstance(m, axonal_recdel):
            m.forward_version = FORWARD_VERSION
            n_ax += 1
    # Every kernel honors round_delays, but assert the flag actually reached the
    # layers: a silently-False STE run would be an exact duplicate of a no-round run.
    ste_on = [m.round_delays for m in _recdel_layers(model)]
    assert all(v == knobs['round_delays'] for v in ste_on), \
        f"round_delays did not reach the recdel layers: {ste_on}"

    print(f"\n=== {model_name} | mode={mode} | delay_std_init={dsi} | seed={seed} ===")
    print(f"forward_version: '{FORWARD_VERSION}' on {n_ax} axonal layers, "
          f"'{SYN_FORWARD_VERSION}' on {n_syn} synaptic layers")
    print(f"rounding: round_pos_each_epoch={knobs['round_pos']}, "
          f"round_mode='{knobs['round_mode']}', round_delays(STE)={knobs['round_delays']}")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    # Run directory for this run.
    run_dir = os.path.join(sweep_root, model_name, tag, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2)

    # Initial (post-init, pre-training) recurrent delays per layer -> "beginning".
    recdel_layers = _recdel_layers(model)
    init_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                   for i, l in enumerate(recdel_layers)}
    np.savez(os.path.join(run_dir, 'init_delays.npz'), **init_delays)

    # Reference for per-epoch delay-variation tracking.
    prev_recurrent_delays_vals = [l.recurrent_delays.detach().clone()
                                  for l in recdel_layers]

    # For storing results
    train_res = pd.DataFrame()
    val_res = pd.DataFrame()
    # rows = layer{i}, cols = epoch . Mean-abs delay variation per epoch
    delay_var_res = pd.DataFrame()
    best_val_acc = 0.0

    for epoch in range(config.epochs):

        # If update sigma at each epoch :
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config)
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)

        for sc in scheduler:
            sc.step()

        # for logs
        train_res[str(epoch)] = [train_acc, train_loss]
        val_res[str(epoch)] = [val_acc, val_loss]
        train_res.to_csv(os.path.join(run_dir, 'train_res.csv'), index=True)
        val_res.to_csv(os.path.join(run_dir, 'val_res.csv'), index=True)

        # Following delay variation (per recdel layer)
        epoch_var = {}
        for rec_idx, m in enumerate(recdel_layers):
            current = m.recurrent_delays.detach()
            prev = prev_recurrent_delays_vals[rec_idx]
            delta = (current - prev).abs().mean().item()
            epoch_var[f'layer{rec_idx}'] = delta
            prev_recurrent_delays_vals[rec_idx] = current.clone()
        delay_var_res[str(epoch)] = pd.Series(epoch_var)
        delay_var_res.to_csv(os.path.join(run_dir, 'delay_variation.csv'), index=True)

        state = {'net': model.state_dict(), 'acc': val_acc, 'epoch': epoch}
        torch.save(state, os.path.join(run_dir, 'last.pth'))

        if val_acc >= best_val_acc:
            torch.save(state, os.path.join(run_dir, 'best.pth'))
            best_val_acc = val_acc

        print(
            'Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, acc: {:.4f}, best: {:.4f}'
            .format(epoch, config.epochs,
                    optimizer[0].param_groups[0]['lr'],
                    optimizer[1].param_groups[0]['lr'],
                    val_acc, best_val_acc))

    ### Testing the best model ###
    best_ckpt = torch.load(os.path.join(run_dir, 'best.pth'))
    model.load_state_dict(best_ckpt['net'])
    best_epoch = best_ckpt['epoch']
    best_val = best_ckpt['acc']

    # Under 'stochastic', best.pth was saved right after a test() that already
    # projected the delays onto integers, so this final test's stochastic round sees
    # frac = 0 and is an exact no-op -> the reported accuracy carries no sampling
    # noise. Under 'ste' the forward rounds every step anyway.
    final_test_acc, final_test_loss = test(test_loader, model, best_epoch, device, config)

    # Delays as actually used at test time, for the downstream analyses.
    final_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                    for i, l in enumerate(_recdel_layers(model))}
    np.savez(os.path.join(run_dir, 'final_delays.npz'), **final_delays)

    print(f"\nFinal Test ({model_name}, {tag}, seed {seed}) "
          f"Acc: {final_test_acc:.4f}, Loss: {final_test_loss:.4f} "
          f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")


    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        json.dump({"acc": final_test_acc, "loss": final_test_loss,
                   "best_val": best_val, "best_epoch": int(best_epoch),
                   "delay_std_init": dsi, "mode": mode,
                   "round_pos_each_epoch": knobs['round_pos'],
                   "round_mode": knobs['round_mode'],
                   "round_delays": knobs['round_delays']}, fid, indent=2)


    return {"model": model_name, "mode": mode, "delay_std_init": dsi, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_acc": final_test_acc}


if __name__ == "__main__":

    # Sweep axes (all overridable via env for splitting / smoke tests).
    DELAY_STD_INITS = [int(s) for s in os.environ.get("HAR_DELAY_STD_INITS", "3,8,13,18,23").split(",")]
    MODELS = [s for s in os.environ.get(
        "HAR_MODELS", "SNN_recurrent_delays,SNN_synaptic_recurrent_delays").split(",") if s]
    seed_list = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    # The two alternatives under test. 'nearest' is available but left out by default:
    # exp/HAR/std_init_sweep/ already holds that baseline.
    ROUND_MODES = [s for s in os.environ.get("HAR_ROUND_MODES", "ste,stochastic").split(",") if s]
    SWEEP_ROOT = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/rounding_alternatives_sweep')
    EPOCHS_OVERRIDE = os.environ.get("HAR_EPOCHS", "")
    # optional batch-size override. Synaptic (N,N) delays are memory-heavy at large
    # delay_std_init, so a smaller batch avoids CUDA OOM.
    BATCH_OVERRIDE = os.environ.get("HAR_BATCH_SIZE", "")

    unknown = [m for m in ROUND_MODES if m not in MODES]
    if unknown:
        raise SystemExit(f"Unknown rounding mode(s) {unknown}, expected from {sorted(MODES)}.")

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {SWEEP_ROOT}")
    print(f"Models: {MODELS} | modes: {ROUND_MODES} | delay_std_init: {DELAY_STD_INITS} "
          f"| seeds: {seed_list}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    manifest = {
        "sweep_root": SWEEP_ROOT,
        "models": MODELS,
        "round_modes": ROUND_MODES,
        "delay_std_inits": DELAY_STD_INITS,
        "seeds": seed_list,
        "forward_version": FORWARD_VERSION,
        "syn_forward_version": SYN_FORWARD_VERSION,
        "runs": [],
    }
    # (model, mode, dsi) -> list of final test accuracies, for the end-of-sweep summary
    group_acc = {}

    for model_name in MODELS:
        for mode in ROUND_MODES:
            for dsi in DELAY_STD_INITS:
                for run_seed in seed_list:
                    row = run_single(model_name, dsi, run_seed, SWEEP_ROOT, device,
                                     mode=mode, epochs_override=EPOCHS_OVERRIDE,
                                     batch_override=BATCH_OVERRIDE)

                    manifest["runs"].append(row)
                    with open(os.path.join(SWEEP_ROOT, 'manifest_rounding_alternatives.json'), 'w') as fid:
                        json.dump(manifest, fid, indent=2)

                    group_acc.setdefault((model_name, mode, dsi), []).append(row["final_test_acc"])

    print("\n==================== SWEEP SUMMARY ====================")
    for (model_name, mode, dsi), accs in sorted(group_acc.items()):
        accs = np.array(accs)
        print(f"{model_name:34s} {mode:11s} std={dsi:<3d} "
              f"test acc = {accs.mean():.4f} +/- {accs.std():.4f}  (seeds={list(seed_list)})")
    print(f"\nManifest: {os.path.join(SWEEP_ROOT, 'manifest_rounding_alternatives.json')}")
