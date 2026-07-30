"""Weight-space lesion: remove recurrent branches by delay threshold.

Zeroes the recurrent branches whose delay is below or above a threshold,
evaluates the clean test set, then restores. Companion to the permutation lesion:
that one shuffles delays and keeps every weight, this one removes weights, so
together they separate "the delay values matter" from "these branches matter".

Deterministic. Reports the removed weight fraction, since a fixed threshold prunes
different fractions in different families.
"""

import torch

from common.delay_perturb import _recdel_modules


# --------------------------------------------------------------------------- #
# Recurrent weight access
# --------------------------------------------------------------------------- #
def snapshot_weights(model):
    """Detached clones of every recurrent-weight tensor (to restore after a prune)."""
    return [m.recurrent_weights.detach().clone() for m in _recdel_modules(model)]


def restore_weights(model, snapshot):
    with torch.no_grad():
        for m, w0 in zip(_recdel_modules(model), snapshot):
            m.recurrent_weights.copy_(w0)


# --------------------------------------------------------------------------- #
# Delay-threshold mask
# --------------------------------------------------------------------------- #
def _keep_mask(delays, direction, threshold):
    """Boolean mask over ``recurrent_weights`` (shape ``(N_in, N_out)``) that is
    ``True`` where the branch is kept. ``delays`` is rounded to the integer eval
    regime before thresholding. ``direction`` selects which end to remove:
    ``"short"`` removes ``delay < threshold``. ``"long"`` removes ``delay > threshold``."""
    d = delays.detach().round()
    remove = d < threshold if direction == "short" else d > threshold  # per-branch
    if d.ndim == 1:                       # axonal: (N_out,) -> broadcast over rows
        remove = remove.unsqueeze(0)      # (1, N_out), zeroes whole columns
    return ~remove


def pruned_fraction(model, direction, threshold, layer_index=None):
    """Fraction of recurrent weight entries zeroed by this threshold (reported in
    the console/CSV. The x-axis is ``threshold`` itself). ``layer_index`` (0-based
    over the recurrent-delay modules) restricts the count to that one layer. ``None``
    counts every layer (the global prune)."""
    kept, total = 0, 0
    for li, m in enumerate(_recdel_modules(model)):
        if layer_index is not None and li != layer_index:
            continue
        keep = _keep_mask(m.recurrent_delays, direction, threshold)
        keep = keep.expand_as(m.recurrent_weights)
        kept += int(keep.sum().item())
        total += keep.numel()
    return 1.0 - kept / total if total else 0.0


# --------------------------------------------------------------------------- #
# Branch-pruned evaluation (clean input, masked weights, restored after)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_branch_pruned(loader, model, device, calc_metric, direction, threshold,
                           input_transform=None, layer_index=None):
    """Test-set accuracy (%) with the recurrent branches selected by ``(direction,
    threshold)`` removed (their ``recurrent_weights`` zeroed). Mirrors the trainers'
    ``test()`` loop.
    """
    from delrec.utils import reset_states

    snap = snapshot_weights(model)
    with torch.no_grad():
        for li, (m, w0) in enumerate(zip(_recdel_modules(model), snap)):
            if layer_index is not None and li != layer_index:
                continue
            keep = _keep_mask(m.recurrent_delays, direction, threshold).to(w0.dtype)
            m.recurrent_weights.copy_(w0 * keep)
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
        restore_weights(model, snap)


# Registry: direction -> human-readable description. The magnitude-axis label is
# shared (the threshold d in timesteps).
DIRECTIONS = {
    "short": "Prune branches with delay < d",
    "long":  "Prune branches with delay > d",
}
THRESHOLD_LABEL = "Delay threshold d (timesteps)"
