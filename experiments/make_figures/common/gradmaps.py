"""Dataset-agnostic input-gradient temporal maps for recurrent-delay SNNs.

For each model it back-propagates a class-readout loss to a synthetic input and
visualizes d(readout)/d(input) over time, a "gradient map" showing which input
timesteps each class relies on, and the temporal energy profile
E(t) = sum over input channels of the squared gradient (a proxy for the model's
effective memory horizon).

The dataset packages supply the run set, the ``Config`` instance and the class
list. The compute + plotting below are shared. Input dimensionality is read off
``cfg.input_size`` so this works for SSC (700-channel spikes) and HAR (3-channel
gyro) alike.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import first: inserts the repo root on sys.path so ``delrec`` resolves below.
from common.utils import load_model
from common import style   # importing applies the DelRec theme (rcParams)

from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import surrogate

from delrec.utils import reset_states


## Loaders

def load_gradmap_model(model_class_name, run_dir, ckpt_name, device, config,
                       forward_version=None):
    """Rebuild a model for gradient analysis: smooth ATan surrogate (so the input gradient is
    meaningful), sigma=0 and DCLS SIG=0 (sharp single-tap delays), rounded delay positions.
    """
    ckpt_path = str(Path(run_dir) / ckpt_name) if run_dir is not None else None
    model, cfg, _ = load_model(
        model_class_name, ckpt_path, device, config,
        surrogate_fn=surrogate.ATan(alpha=2.0),
        sigma_zero=True, zero_dcls_sig=True, round_positions=True,
        strict=False, verbose=True, forward_version=forward_version,
    )
    return model, cfg


def preload_samples(cfg, target_classes, split="test"):
    from delrec.datasets import load_dataset
    loaders = load_dataset(cfg)
    loader = {"train": loaders[0], "valid": loaders[1], "test": loaders[2]}[split]
    needed = set(target_classes)
    results = {}
    if hasattr(loader, 'reset'):
        loader.reset()
    for batch in loader:
        x, y = batch[0], batch[1]
        for cls in list(needed):
            match = (y.cpu() == cls).nonzero(as_tuple=False).view(-1)
            if match.numel():
                b = int(match[0])
                if x.dim() == 3:
                    x_TN = x[b] if x.shape[0] == y.shape[0] else x[:, b]
                elif x.dim() == 2:
                    x_TN = x[b].unsqueeze(0)
                else:
                    continue
                results[cls] = x_TN.float().cpu()
                needed.discard(cls)
        if not needed:
            break
    if needed:
        raise RuntimeError(f"Classes not found: {needed}")
    return results


## Helper

def _disable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, sj_layer.Dropout):
            m.eval()


## Computations

def _make_synthetic_input(mode, T, B, cfg, dev, impulse_value):
    if mode == "zeros":
        return torch.zeros(T, B, cfg.input_size, device=dev, dtype=torch.float32)
    elif mode == "impulse":
        data = torch.zeros(T, B, cfg.input_size, device=dev, dtype=torch.float32)
        data[0, :, :] = impulse_value
        return data
    elif mode == "ones":
        return torch.full((T, B, cfg.input_size), impulse_value, device=dev, dtype=torch.float32)
    elif mode == "noise":
        return 0.01 * torch.randn(T, B, cfg.input_size, device=dev, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


@torch.enable_grad()
def compute_gradmaps_batched(
    model: torch.nn.Module,
    cfg,
    T: int,
    dev: torch.device,
    target_classes_list: list,
    mode: str = "zeros",
    impulse_value: float = 0.0,
    loss_point: str = "final",
) -> dict:
    def _prep_model():
        model.train()
        _disable_dropout(model)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
        if hasattr(model, 'round_pos'):
            model.round_pos()

    B = len(target_classes_list)
    _prep_model()
    model.zero_grad(set_to_none=True)
    reset_states(model)
    data = _make_synthetic_input(mode, T, B, cfg, dev, impulse_value)
    x = data.detach().requires_grad_(True)
    print(f"Gradmap | input={mode} | loss_point={loss_point} | x.shape={tuple(x.shape)}")
    out = model(x)
    if loss_point == "sum":
        loss = sum(-out[:, b, c].sum() for b, c in enumerate(target_classes_list))
    elif loss_point == "final":
        loss = sum(-out[-1, b, c].sum() for b, c in enumerate(target_classes_list))
    else:
        raise ValueError(f"Unknown loss_point: {loss_point!r}")
    loss.backward()
    grad = -x.grad.detach()
    return {cls: grad[:, b] for b, cls in enumerate(target_classes_list)}


# Plots

def crop_gradmaps(gradmaps_dict, t_start):
    """Crop {cls: Tensor(T, N_in)} to [t_start:]."""
    if t_start <= 0:
        return gradmaps_dict
    return {cls: g[t_start:] for cls, g in gradmaps_dict.items()}


def plot_gradmap_comparison(
    grads_list: list,
    titles: list,
    clip_q: float = 0.99,
    ncols: int = 3,
) -> plt.Figure:
    mats = [g.detach().T.cpu().numpy() for g in grads_list]
    n = len(mats)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols + 1.0, 5.0 * nrows),
                             squeeze=False)

    for i, (mat, title) in enumerate(zip(mats, titles)):
        ax = axes[i // ncols, i % ncols]
        mscale = float(np.quantile(np.abs(mat), clip_q)) if clip_q else float(np.max(np.abs(mat)))
        mscale = mscale if mscale > 0 else 1.0
        im_grad = ax.imshow(np.clip(mat, -mscale, mscale), aspect="auto", origin="lower",
                            cmap=style.DIVERGING, vmin=-mscale, vmax=mscale)
        ax.set(title=title, xlabel="Time", ylabel="Input neuron")
        fig.colorbar(im_grad, ax=ax, fraction=0.03, pad=0.02)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout()
    return fig


def plot_gradmap_paper_row(
    grads_list: list,
    labels: list,
    clip_q: float = 0.99,
    cbar_label: str = "Input gradient",
    y_label: str = "Input channel",
    panel_w: float = 2.0,
    own_scale: list = None,
) -> plt.Figure:
    """A paper-ready row of gradient maps, meant to be dropped into a figure as a sub-panel:
    print-size text, panel letters, one shared symmetric color scale and a single
    colorbar, and an x axis in steps before the readout (0 = the step the loss is taken
    at), which is crop-independent unlike a raw index.
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    mats = [g.detach().T.cpu().numpy() for g in grads_list]
    n = len(mats)
    own = list(own_scale) if own_scale is not None else [False] * n

    def _sym_scale(idx):
        pooled = np.abs(np.concatenate([mats[i].ravel() for i in idx]))
        s = float(np.quantile(pooled, clip_q)) if clip_q else float(pooled.max())
        return float(f"{s:.2g}") if s > 0 else 1.0  # round ends: 6.43e-3 -> 6.4e-3

    # Color-scale groups: the shared panels form one, each own-scale panel its
    # own. Ordered by rightmost panel, so each colorbar sits right of its group.
    shared = [i for i in range(n) if not own[i]]
    groups = ([(shared, _sym_scale(shared), False)] if shared else []) + \
             [([i], _sym_scale([i]), True) for i in range(n) if own[i]]
    groups.sort(key=lambda g: max(g[0]))
    scale_of = {i: s for idx, s, _ in groups for i in idx}

    with plt.rc_context(style.RC_PAPER):
        fig, axes = plt.subplots(
            1, n, figsize=(panel_w * n + 1.0 + 0.9 * (len(groups) - 1), 2.1),
            sharey=True, constrained_layout=True, squeeze=False)
        axes = axes[0]
        for i, (ax, mat, label) in enumerate(zip(axes, mats, labels)):
            T = mat.shape[1]
            scale = scale_of[i]
            ax.imshow(np.clip(mat, -scale, scale), aspect="auto", origin="lower",
                      cmap=style.DIVERGING, vmin=-scale, vmax=scale,
                      interpolation="nearest",
                      extent=(-T + 0.5, 0.5, -0.5, mat.shape[0] - 0.5))
            # A heatmap reads as a framed field, so put the box back (RC drops the
            # top/right spines for line plots).
            for sp in ax.spines.values():
                sp.set_visible(True)
            ax.tick_params(direction="out")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
            ax.set_xlabel("Steps before readout")
            ax.set_title(label, loc="left", x=0.075, pad=4)
            ax.annotate(f"{chr(ord('a') + i)}", xy=(0, 1), xytext=(0, 4),
                        xycoords="axes fraction", textcoords="offset points",
                        fontweight="bold", fontsize=plt.rcParams["axes.titlesize"],
                        color=style.INK_PRIMARY, ha="left", va="bottom")
        axes[0].set_ylabel(y_label)

        for idx, scale, is_own in groups:
            cbar = fig.colorbar(axes[idx[0]].images[0], ax=[axes[i] for i in idx],
                                fraction=0.045, pad=0.02, aspect=18)
            cbar.set_label(f"{cbar_label} (own scale)" if is_own else cbar_label,
                           color=style.INK_SECONDARY)
            cbar.set_ticks([-scale, 0, scale])
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            cbar.formatter = fmt
            cbar.ax.yaxis.set_offset_position("left")
            cbar.update_ticks()
            cbar.outline.set_linewidth(0.6)
            cbar.outline.set_edgecolor(style.BASELINE)
            cbar.ax.tick_params(width=0.6, length=2.5)
    return fig


def plot_gradmap_energy_profiles(all_gradmaps, target_classes, out_dir,
                                 normalize=False) -> list:
    suffix = "_normalized" if normalize else ""
    out_dir = Path(out_dir)
    tags = list(all_gradmaps.keys())
    n_cls = len(target_classes)

    ncols = min(n_cls, 4)
    nrows = max(1, (n_cls + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    T_global = None
    for ci, cls in enumerate(target_classes):
        ax = axes_flat[ci]
        for tag in tags:
            if cls not in all_gradmaps[tag]:
                continue
            entry = all_gradmaps[tag][cls]

            if isinstance(entry, list):
                arrays = [G.detach().cpu().numpy() if isinstance(G, torch.Tensor) else np.asarray(G)
                          for G in entry]
            else:
                G = entry.detach().cpu().numpy() if isinstance(entry, torch.Tensor) else np.asarray(entry)
                arrays = [G]

            if normalize:
                arrays = [g / (np.linalg.norm(g) + 1e-12) for g in arrays]

            E_seeds = [np.sum(g ** 2, axis=1) for g in arrays]
            T = E_seeds[0].shape[0]
            T_global = T
            E_stack = np.stack(E_seeds, axis=0)
            E_mean = E_stack.mean(axis=0)
            E_sem = (E_stack.std(axis=0, ddof=1) / np.sqrt(E_stack.shape[0])
                     if E_stack.shape[0] > 1 else np.zeros(T))

            if E_mean.sum() < 1e-12:
                continue

            t_idx = np.arange(T, dtype=np.float64)
            centroid = np.sum(t_idx * E_mean) / E_mean.sum()

            (line,) = ax.plot(E_mean, label=tag, lw=1.2)
            ax.fill_between(t_idx, E_mean - E_sem, E_mean + E_sem,
                            alpha=0.25, color=line.get_color())
            ax.axvline(centroid, ls="--", lw=0.8, alpha=0.7, color=line.get_color())

        if T_global is not None:
            ax.axvline(T_global / 2, color="gray", ls=":", lw=0.8, label="T/2")
        ylabel = "Normalized Energy E(t)" if normalize else "Energy E(t)"
        ax.set(title=f"Class {cls}", xlabel="Timestep", ylabel=ylabel)
        ax.legend(fontsize=6)

    for ci in range(n_cls, len(axes_flat)):
        axes_flat[ci].set_visible(False)

    title = "Gradient Energy Temporal Profiles"
    if normalize:
        title += " (Frobenius-normalized)"
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig_paths = [(fig, out_dir / f"gradmap_energy_profiles{suffix}.svg")]

    # Overall energy profile: mean over all classes and seeds
    fig_all, ax_all = plt.subplots(figsize=(7, 4))
    for tag in tags:
        all_E = []
        for cls in target_classes:
            if cls not in all_gradmaps[tag]:
                continue
            entry = all_gradmaps[tag][cls]
            if isinstance(entry, list):
                arrays = [G.detach().cpu().numpy() if isinstance(G, torch.Tensor)
                          else np.asarray(G) for G in entry]
            else:
                G = (entry.detach().cpu().numpy() if isinstance(entry, torch.Tensor)
                     else np.asarray(entry))
                arrays = [G]
            if normalize:
                arrays = [g / (np.linalg.norm(g) + 1e-12) for g in arrays]
            for g in arrays:
                E = np.sum(g ** 2, axis=1)
                if E.sum() > 1e-12:
                    all_E.append(E)

        if not all_E:
            continue

        E_stack = np.stack(all_E, axis=0)
        E_mean = E_stack.mean(axis=0)
        E_sem = (E_stack.std(axis=0, ddof=1) / np.sqrt(E_stack.shape[0])
                 if E_stack.shape[0] > 1 else np.zeros(E_mean.shape[0]))
        T_len = E_mean.shape[0]
        # x in "steps before readout" (0 = the step the loss is taken at), the
        # same axis as the per-seed gradmaps.
        t_idx = np.arange(T_len, dtype=np.float64) - (T_len - 1)
        centroid = np.sum(t_idx * E_mean) / E_mean.sum()

        (line,) = ax_all.plot(t_idx, E_mean, label=tag, lw=1.2)
        ax_all.fill_between(t_idx, E_mean - E_sem, E_mean + E_sem,
                            alpha=0.25, color=line.get_color())
        ax_all.axvline(centroid, ls="--", lw=0.8, alpha=0.7, color=line.get_color())

    if T_global is not None:
        ax_all.axvline(-(T_global - 1) / 2, color="gray", ls=":", lw=0.8, label="T/2")
    ylabel_all = "Normalized Energy E(t)" if normalize else "Energy E(t)"
    ax_all.set(title="Mean over Classes & Seeds", xlabel="Steps before readout",
               ylabel=ylabel_all)
    ax_all.legend(fontsize=8)
    fig_all.tight_layout()
    fig_paths.append((fig_all, out_dir / f"gradmap_energy_profiles_overall{suffix}.svg"))

    return fig_paths
