"""Sweep over the firing-rate penalty lambda of Eq. (20).

Produces: the accuracy/energy frontier.
Outputs:  trained_models/HAR/spike_penalty_sweep/
Usage:    python experiments/sweeps/sweep_HAR_spikepenalty.py

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
import os
import numpy as np

from configs.perf_HAR import Config

from delrec.training.har import *
from delrec.networks import *
import delrec.networks as snn_module
from delrec.datasets import load_dataset


# ── Recurrent kernel choice (hardcode here. None = library default) ────────────
# Library default on GPU: axonal -> 'triton_exact', synaptic -> 'eventdriven'
# (both exact vs v2). Set to force a kernel, e.g. AXONAL_FORWARD_VERSION =
# 'eventdriven'. Env REC_FWD / SYN_REC_FWD take precedence if set.
AXONAL_FORWARD_VERSION = os.environ.get("REC_FWD") or None
SYN_FORWARD_VERSION    = os.environ.get("SYN_REC_FWD") or None

# ── Fixed delay_std_init for the sweep (hardcode here. Env HAR_DELAY_STD_INIT wins) ─
# Every run trains at this delay_std_init (the spike-penalty axis is the swept one).
DELAY_STD_INIT = int(os.environ.get("HAR_DELAY_STD_INIT", "3"))

# Spike-penalty sweep for HAR, built on sweep_HAR_weightdecay.py.
# For every (model, spike_penalty, seed) triple it trains the HAR pipeline at a
# fixed delay_std_init (default 3) and NO weight decay on the delays, while
# applying an L2 firing-rate penalty to every layer's spikes: the loss gets
# ``config.spike_penalty * get_spike_cost(model)`` (get_spike_cost reads the
# spike_registrator modules, one after each hidden layer. See src/HAR/trainer.py).
# Sweeping spike_penalty traces how test accuracy degrades as the network is
# pushed to fire fewer spikes (an energy/accuracy trade-off, per delay condition).
#
# The recurrent kernel is left at its DEFAULT: the spike-sparse 'eventdriven' Triton
# scan on CUDA (HAR has decay_input=False, so it qualifies) and pure-torch 'v2' on
# CPU, forward_version is intentionally NOT overridden here.
#
# Per run it persists:
#   config.json, train_res.csv, val_res.csv, best.pth, last.pth   (as perf_HAR)
#   init_delays.npz, per-layer recurrent_delays right after init
#   final_delays.npz, per-layer recurrent_delays of the tested (best, rounded) model
#   delay_variation.csv, per-epoch mean-abs delay variation per recdel layer
#   final_test.json, final test acc/loss + best val/epoch + spike_penalty +
#                           firing_rate/spike_cost of the tested model
# A manifest.json at the sweep root indexes all runs for the plotting/analysis script.
#
# ``run_single`` trains ONE triple and returns its manifest row. It is reused by the
# sequential loop below and by parallel_sweep_HAR_spikepenalty.py (one process per run).


def _recdel_layers(model):
    """Return the axonal_recdel (incl. synaptic_recdel) layers in model.layers order."""
    return [m for m in model.layers if isinstance(m, axonal_recdel)]


def _lam_tag(lam):
    """Filesystem-safe, stable directory tag for a spike-penalty value: lam0, lam0.1, lam3."""
    return f"lam{lam:g}"


def _measure_spikes(model, loader, device):
    """One forward pass over the test set (no grad, no penalty applied to the loss).
    Returns (firing_rate, spike_cost): the mean spikes-per-neuron-per-timestep over
    all spike_registrator modules and get_spike_cost's L2 cost, averaged over batches.
    Mirrors test()'s permute/reset loop and analysis/common/efficiency.evaluate_efficiency."""
    registrators = [m for m in model.modules() if isinstance(m, spike_registrator)]
    model.eval()
    spk_sum, spk_elems = 0.0, 0
    costs = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
            reset_states(model=model)
            model(inputs)
            costs.append(get_spike_cost(model).item())
            for reg in registrators:
                s = reg.spikes
                spk_sum += s.sum().item()
                spk_elems += s.numel()
    firing_rate = spk_sum / max(spk_elems, 1)
    spike_cost = float(np.mean(costs)) if costs else 0.0
    return firing_rate, spike_cost


def run_single(model_name, spike_penalty, seed, sweep_root, device, *,
               delay_std_init=3, round_pos=True, epochs_override="", batch_override="", subdir=None):
    """Train a single run and persist all artifacts under
    ``sweep_root/model_name/<subdir>/seed<seed>/``.
    """
    tag = subdir if subdir is not None else _lam_tag(spike_penalty)
    config = Config()
    config.seed = seed
    config.delay_std_init = delay_std_init
    config.weight_decay_positions = 0.0
    config.spike_penalty = spike_penalty
    config.round_pos_each_epoch = round_pos
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = getattr(snn_module, model_name)(config).to(device)

    # Apply the hardcoded kernel choice per layer type (None -> library default:
    # axonal triton_exact / synaptic eventdriven on CUDA, v2 on CPU).
    from delrec.delay_layers import synaptic_recdel
    for _layer in model.layers:
        if isinstance(_layer, synaptic_recdel):              # check subclass first
            if SYN_FORWARD_VERSION:
                _layer.forward_version = SYN_FORWARD_VERSION
        elif isinstance(_layer, axonal_recdel):
            if AXONAL_FORWARD_VERSION:
                _layer.forward_version = AXONAL_FORWARD_VERSION
    n_recdel = len(_recdel_layers(model))
    print(f"\n=== {model_name} | lam={spike_penalty:g} | std{delay_std_init} | seed={seed} ===")
    print(f"recurrent layers: {n_recdel} | axonal={AXONAL_FORWARD_VERSION or 'default(triton_exact)'} "
          f"synaptic={SYN_FORWARD_VERSION or 'default(eventdriven)'}")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    # Run directory for this run.
    run_dir = os.path.join(sweep_root, model_name, tag, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2)

    # Initial (post-init, pre-training) recurrent delays per layer.
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

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config,
                                      penalize_spikes=True)
        val_acc, val_loss = test(valid_loader, model, epoch, device, config,
                                 penalize_spikes=True)

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

    # Final test at sigma=0 (sharp single-tap delays), rounded, matches perf_HAR.
    for m in model.modules():
        if isinstance(m, axonal_recdel):
            m.sigma = 0.0
            m.use_sig_p = False
    if getattr(config, 'round_pos_each_epoch', False) and hasattr(model, 'round_pos'):
        model.round_pos()

    final_test_acc, final_test_loss = test(test_loader, model, best_epoch, device, config)

    # Firing rate + spike cost of the exact tested model (clean, no penalty in the loss).
    firing_rate, spike_cost = _measure_spikes(model, test_loader, device)

    # Delays of the exact model that was tested (best, sigma=0, rounded).
    final_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                    for i, l in enumerate(_recdel_layers(model))}
    np.savez(os.path.join(run_dir, 'final_delays.npz'), **final_delays)

    print(f"\nFinal Test ({model_name}, {tag}, seed {seed}) "
          f"Acc: {final_test_acc:.4f}, Loss: {final_test_loss:.4f} "
          f"| firing_rate={firing_rate:.5f} spike_cost={spike_cost:.5f} "
          f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")


    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        json.dump({"acc": final_test_acc, "loss": final_test_loss,
                   "best_val": best_val, "best_epoch": int(best_epoch),
                   "spike_penalty": spike_penalty,
                   "firing_rate": firing_rate, "spike_cost": spike_cost,
                   "delay_std_init": delay_std_init}, fid, indent=2)


    return {"model": model_name, "spike_penalty": spike_penalty,
            "delay_std_init": delay_std_init, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_acc": final_test_acc, "firing_rate": firing_rate}


if __name__ == "__main__":

    # Sweep axes (all overridable via env for splitting / smoke tests).
    # Calibrated on the trained std3 models: spike_cost ~0.14, CE loss ~0.5, so the
    # penalty term is lam*0.14 -> lam 0.1 negligible, 1 meaningful, 3 ~CE, 10-30 collapse.
    SPIKE_PENALTIES = [float(s) for s in os.environ.get(
        "HAR_SPIKE_PENALTIES", "0,0.1,0.3,1,3,10,30").split(",")]
    MODELS = [s for s in os.environ.get(
        "HAR_MODELS",
        "SNN_recurrent_delays,SNN_synaptic_recurrent_delays,"
        "SNN_fixed_recurrent_delays,SNN_fixed_synaptic_recurrent_delays,"
        "SNN_vanilla_recurrent").split(",") if s]
    seed_list = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    # DELAY_STD_INIT hardcoded near the top of the file (env HAR_DELAY_STD_INIT wins).
    # round_pos=1 matches the existing std comparison runs (roundpos1).
    ROUND_POS = os.environ.get("HAR_ROUND_POS", "1").lower() in ("1", "true", "yes")
    SWEEP_ROOT = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/spike_penalty_sweep')
    # optional epoch override (handy for range-finding / smoke tests). Defaults to Config.epochs
    EPOCHS_OVERRIDE = os.environ.get("HAR_EPOCHS", "")
    # optional batch-size override. Synaptic (N,N) delays are memory-heavier.
    BATCH_OVERRIDE = os.environ.get("HAR_BATCH_SIZE", "")

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {SWEEP_ROOT}")
    print(f"Models: {MODELS} | delay_std_init: {DELAY_STD_INIT} | "
          f"spike_penalties: {SPIKE_PENALTIES} | seeds: {seed_list} | round_pos: {ROUND_POS}")

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
        "delay_std_init": DELAY_STD_INIT,
        "spike_penalties": SPIKE_PENALTIES,
        "seeds": seed_list,
        "round_pos": ROUND_POS,
        "runs": [],
    }
    # (model, lam) -> list of final test accuracies, for the end-of-sweep summary
    group_acc = {}

    for model_name in MODELS:
        for lam in SPIKE_PENALTIES:
            for run_seed in seed_list:
                row = run_single(model_name, lam, run_seed, SWEEP_ROOT, device,
                                 delay_std_init=DELAY_STD_INIT, round_pos=ROUND_POS,
                                 epochs_override=EPOCHS_OVERRIDE, batch_override=BATCH_OVERRIDE)

                manifest["runs"].append(row)
                with open(os.path.join(SWEEP_ROOT, 'manifest.json'), 'w') as fid:
                    json.dump(manifest, fid, indent=2)

                group_acc.setdefault((model_name, lam), []).append(row["final_test_acc"])

    print("\n==================== SWEEP SUMMARY ====================")
    for (model_name, lam), accs in sorted(group_acc.items()):
        accs = np.array(accs)
        print(f"{model_name:32s} lam={lam:<8g} "
              f"test acc = {accs.mean():.4f} +/- {accs.std():.4f}  (seeds={list(seed_list)})")
    print(f"\nManifest: {os.path.join(SWEEP_ROOT, 'manifest.json')}")
