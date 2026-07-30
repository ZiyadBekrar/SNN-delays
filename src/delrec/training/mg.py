"""Training and evaluation loop for Mackey-Glass, the one regression task.

Returns NMSE / MSE / NRMSE / R^2 rather than accuracy, and reduces the readout
with a leaky-integrator EMA (``readout_MG``) instead of a mean over time: the
target lies past the end of the window, so the last steps carry the signal.
Reuses the SSC ``init_optim_sche`` unchanged.
"""

import numpy as np
import torch

from delrec.training.ssc import init_optim_sche  # Adam + cosine over (weights, positions)
from delrec.utils import no_log, calc_loss_MG, readout_MG, progress_bar, reset_states

# Mackey-Glass is a regression task, so unlike the classification trainers (which
# report accuracy) train/evaluate report the MSE and its normalized forms:
#   NMSE  = MSE / var(targets)   (1.0 == predicting the mean. Lower is better)
#   NRMSE = sqrt(NMSE)
#   R2    = 1, SS_res / SS_tot
# init_optim_sche is reused as-is from the SSC trainer: it already splits the two
# parameter groups (weights vs recurrent_delays) with their own LRs and schedulers.


def compute_metrics(preds, tgts):
    """Regression metrics for two 1-D numpy arrays (a whole split, not a batch,
    NMSE/R2 need the variance of the split's targets)."""
    residuals = preds - tgts
    mse = float(np.mean(residuals ** 2))
    var = float(np.var(tgts))
    nmse = mse / var if var > 0 else float('inf')
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((tgts - np.mean(tgts)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('-inf')
    return {'mse': mse, 'nmse': nmse, 'nrmse': nmse ** 0.5, 'r2': r2}


def train(train_loader, model, optimizer, epoch, device, config, logger=no_log):
    train_loss = 0
    sq_err = 0.0
    all_tgts = []

    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs of shape (Batch, Time, Neurons)
        inputs = inputs.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
        targets = targets.float().to(device)

        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_MG(outputs, targets, config.tau_readout)

        train_loss += loss.item()
        sq_err += loss.item() * targets.size(0)  # loss is the batch-mean squared error
        all_tgts.append(targets.detach().cpu())

        for opt in optimizer: opt.zero_grad()
        loss.backward()
        for opt in optimizer: opt.step()

        progress_bar(
            batch_idx, len(train_loader), 'MSE: %.6f' % (train_loss / (batch_idx + 1))
        )

    avg_loss = train_loss / len(train_loader)
    tgts = torch.cat(all_tgts).numpy()
    mse = sq_err / len(tgts)
    var = float(np.var(tgts))
    nmse = mse / var if var > 0 else float('inf')

    logger({
        'train/loss': avg_loss,
        'train/nmse': nmse,
        'epoch': epoch,
    })

    logger(model.log_params())

    return nmse, avg_loss


@torch.no_grad()
def evaluate(loader, model, epoch, device, config, split='test', logger=no_log):
    """Full-split evaluation: returns the regression metrics plus the raw predictions
    and targets (kept for the per-run predictions.npz / the report's prediction traces).
    Mirrors the classification trainers' test(): eval mode, delays rounded (or clamped)
    once up front, states reset per batch."""
    model.eval()

    if getattr(config, 'round_pos_each_epoch', False) and hasattr(model, 'round_pos'):
        model.round_pos()
    elif hasattr(model, 'clamp_delays'):
        model.clamp_delays()

    all_preds, all_tgts = [], []
    for inputs, targets in loader:
        # inputs of shape (Batch, Time, Neurons)
        inputs = inputs.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
        targets = targets.float().to(device)

        reset_states(model=model)
        outputs = model(inputs)
        all_preds.append(readout_MG(outputs, config.tau_readout).cpu())
        all_tgts.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_tgts).numpy()

    metrics = compute_metrics(preds, tgts)
    metrics['preds'] = preds
    metrics['tgts'] = tgts

    logger({
        f'{split}/loss': metrics['mse'],
        f'{split}/nmse': metrics['nmse'],
        f'{split}/r2': metrics['r2'],
        'epoch': epoch,
    })

    return metrics
