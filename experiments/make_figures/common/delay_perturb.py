"""Parameter-space lesions of the trained delays.

Snapshots a model's recurrent delays, applies an operator, evaluates the clean
test set, then restores, so a fixed-delay model is the built-in null. Operators
keep the delays integer and non-negative: permute (shuffle a fraction across
neurons or connections), scale (shrink toward zero, the limit being a delay-free
RSNN), noise, and quantize.

Branches on the delay array's shape, so per-neuron and per-connection delays are
both handled.
"""

import copy

import numpy as np

import torch


# --------------------------------------------------------------------------- #
# Delay parameter access
# --------------------------------------------------------------------------- #
def _recdel_modules(model):
    """The recurrent-delay modules of a model, in layer order (axonal_recdel, which
    synaptic_recdel subclasses, so both delay flavours are covered)."""
    from delrec.delay_layers import axonal_recdel
    return [m for m in model.modules() if isinstance(m, axonal_recdel)]


def snapshot_delays(model):
    """Detached clones of every recurrent-delay tensor (to restore after a lesion)."""
    return [m.recurrent_delays.detach().clone() for m in _recdel_modules(model)]


def restore_delays(model, snapshot):
    with torch.no_grad():
        for m, d0 in zip(_recdel_modules(model), snapshot):
            m.recurrent_delays.copy_(d0)


def _write_delays(model, new_delays):
    with torch.no_grad():
        for m, d in zip(_recdel_modules(model), new_delays):
            m.recurrent_delays.copy_(d)


# --------------------------------------------------------------------------- #
# Operators: op(delays_list, gen) -> new integer delays_list (same shapes)
# --------------------------------------------------------------------------- #
def _as_int(d):
    return d.round().clamp_(min=0)


#: How ``permute_op`` chooses which entries to shuffle among themselves. All three
#: touch the same number of entries for a given fraction, and all three preserve the
#: layer's delay multiset exactly, so the delay distribution, mean, std and total
#: recurrent drive are untouched, and learned/fixed families are matched by
#: construction (unlike an absolute delay threshold, which selects different
#: proportions in families whose delay distributions differ).
PERMUTE_SELECTIONS = {
    "random": "a uniformly random subset (unbiased dose-response)",
    "long": "the longest delays (top quantile) -- is the extended tail placed?",
    "short": "the shortest delays (bottom quantile) -- is the short core placed?",
}


def permute_op(frac, selection="random"):
    """Permute a ``frac`` fraction of each delay tensor's entries among themselves."""
    def op(delays, gen):
        out = []
        for d in delays:
            flat = d.reshape(-1).clone()
            n = flat.numel()
            k = int(round(frac * n))
            if k >= 2:
                if selection == "random":
                    sel = torch.randperm(n, generator=gen, device=d.device)[:k]
                else:
                    order = torch.argsort(flat)          # ascending delay
                    sel = order[-k:] if selection == "long" else order[:k]
                perm = sel[torch.randperm(k, generator=gen, device=d.device)]
                flat[sel] = flat[perm]                   # RHS copies first: safe
            out.append(_as_int(flat.reshape(d.shape)))
        return out
    return op


def realized_dose(model, op, seed, device, layer_index=None):
    """Mean |Δdelay| the operator actually induces (its realized dose)."""
    snap = snapshot_delays(model)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    new = op(snap, gen)
    idx = range(len(snap)) if layer_index is None else [layer_index]
    return float(np.mean([ (new[i] - snap[i]).abs().float().mean().item() for i in idx ]))


def scale_op(magnitude):
    """Multiply every delay by ``alpha = 1, magnitude`` (magnitude 1 → all zero)."""
    alpha = 1.0 - magnitude
    def op(delays, gen):
        return [_as_int(d * alpha) for d in delays]
    return op


def noise_op(sigma):
    """Add ``N(0, sigma)`` timesteps to every delay."""
    def op(delays, gen):
        out = []
        for d in delays:
            noise = torch.randn(d.shape, generator=gen, device=d.device) * sigma
            out.append(_as_int(d + noise))
        return out
    return op


def quantize_op(step):
    """Round every delay to the nearest multiple of ``step`` timesteps."""
    def op(delays, gen):
        if step <= 1:
            return [_as_int(d) for d in delays]
        return [_as_int((d / step).round() * step) for d in delays]
    return op


# Registry: name -> (operator factory, magnitude-axis label, clean magnitude).
OPERATORS = {
    "permute":  (permute_op,  "Fraction of delays permuted",      0.0),
    "scale":    (scale_op,    "Delay shrink fraction (1 = zeroed)", 0.0),
    "noise":    (noise_op,    "Delay noise σ (timesteps)",         0.0),
    "quantize": (quantize_op, "Delay quantization step (timesteps)", 1.0),
}


# --------------------------------------------------------------------------- #
# Delay-perturbed evaluation (clean input, lesioned delays, restored after)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_delay_perturbed(loader, model, device, calc_metric, op, seed,
                             input_transform=None, layer_index=None):
    """Test-set accuracy (%) with delay operator ``op`` applied to the model's recurrent
    delays. Mirrors the trainers' ``test()`` loop.
    """
    from delrec.utils import reset_states

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    snap = snapshot_delays(model)
    new = op(snap, gen)
    if layer_index is not None:
        new = [n if i == layer_index else d0
               for i, (d0, n) in enumerate(zip(snap, new))]
    _write_delays(model, new)
    try:
        model.eval()
        if hasattr(loader, "reset"):
            loader.reset()
        correct, total = 0, 0
        for inputs, targets in loader:
            if input_transform is not None:
                inputs = input_transform(inputs)
            inputs = inputs.permute(1, 0, 2).float().to(device)
            targets = targets.to(device)
            reset_states(model=model)
            outputs = model(inputs)
            correct += calc_metric(outputs, targets)
            total += targets.size(0)
        return 100.0 * correct / total
    finally:
        restore_delays(model, snap)
