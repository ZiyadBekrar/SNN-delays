"""How far back in time gradients reach, measured over training.

Trains the delay-free, fixed-delay and learned-delay models from one shared
initialization and, after every epoch, probes the gradient reaching each past
input step on a fixed held-out batch. The probe is a separate forward/backward
pass, so it needs no change to the training loop.

The probe loss is the cross-entropy of the readout at the final step alone: the
training loss sums the readout over all steps, which gives every input step a
direct path to the loss and so would not isolate what flows through the
recurrence.

Outputs:  figures/SSC/gradflow_epochs/
Usage:    python experiments/make_figures/SSC/grad_flow_train.py --plots-only
"""

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np

from common import gradflow as gf
from common import style
from common.utils import save_fig
import runs as ssc
from common import paths

# model key -> (delrec.networks class, label, color, linestyle)
MODELS = {
    "ax_learned": ("SNN_recurrent_delays", "Axonal learned",
                   style.CONDITION_COLORS["learned"], style.CONDITION_LS["learned"]),
    "ax_fixed":   ("SNN_fixed_recurrent_delays", "Axonal fixed",
                   style.CONDITION_COLORS["fixed"], style.CONDITION_LS["fixed"]),
    # The delay mask is centered at 1+d (src/delrec/), so the frozen
    # parameter value 0 is an effective 1-step recurrence, a plain RSNN.
    "vanilla":    ("SNN_vanilla_recurrent", "Vanilla RSNN (d=1)",
                   style.INK_MUTED, ":"),
}
# SSC bins max_time=1.4 s into config.time_window=250 steps.
MS_PER_STEP = 1400.0 / 250.0
DEFAULT_OUT = os.path.join(ssc.OUT_ROOT, "gradflow_epochs")

# Recurrent forward kernel the models are trained/probed with. Default 'triton_exact'
# (also the axonal CUDA default). All kernels are numerically equivalent, see
# src/delrec/. Alternatives: 'v2' (portable), 'eventdriven', None (default).
FORWARD_VERSION = "triton_exact"

# Jacobian probe: how many fixed random unit readout vectors to average ||Jt^T u|| over,
# and the seed that draws them (fixed, so every model/epoch backprops the same vectors,
# that is what makes absolute magnitudes comparable). See common.gradflow.
N_PROBES = 4
PROBE_SEED = 0

# Vanilla-only weight LR. The delay-free RSNN has a tight 1-step recurrent loop and SSC
# runs with use_batch_norm=False (SHD, where this model is stable, has it True), so nothing
# bounds the pre-activations: the membranes leave the Triangle surrogate's support
# (|v-theta| > 1, where the backward is exactly zero) and training dies for whole epochs.
# Gradient clipping alone does not fix it, it bounds the step, not the activations.
# Lowering only the LR keeps the architecture bit-identical to the delay families, so the
# delays remain the single architectural difference this figure compares.
VANILLA_LR_W = 2e-4


# --------------------------------------------------------------------------- #
# Training with the per-epoch gradient probe
# --------------------------------------------------------------------------- #
def balanced_probe_batch(loader, seed=PROBE_SEED):
    """A class-balanced probe batch: ``batch_size // n_classes`` utterances of every keyword."""
    labels = np.asarray(loader.labels_).astype(int)
    classes = np.unique(labels)
    per = loader.batch_size // len(classes)          # 256 // 35 = 7 -> 245 utterances
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], per, replace=False)
                          for c in classes])
    loader.sample_index = idx        # the iterator slices this. The test loader is
    loader.batch_size = len(idx)     # used for nothing else in this script
    loader.counter = 0
    print(f"  probe batch: {len(idx)} utterances, {per} per class x {len(classes)} classes")
    return next(iter(loader))


def run_model(mkey, config, device, epochs):
    """Train one model for ``epochs`` epochs, probing g(t) each epoch on a fixed held-out
    batch. Returns two (epochs × T) matrices: the cross-entropy probe and the unit-vector
    Jacobian probe (``common.gradflow.jacobian_grad_over_time``, scale-comparable)."""
    import delrec.networks as snn_module
    from delrec.delay_layers import axonal_recdel
    from delrec.datasets import load_dataset
    from delrec.utils import seed_everything
    from delrec.training.ssc import train, init_optim_sche


    cls = MODELS[mkey][0]
    if mkey == "vanilla":
        config.lr_w = VANILLA_LR_W        # see VANILLA_LR_W: stability, not tuning
        print(f"  vanilla: lr_w lowered to {config.lr_w} for stability")
    # Reseed right before construction so both families share one init (weights + the
    # half-normal delay draw), isolating the effect of learning the delays.
    seed_everything(seed=config.seed, is_cuda=True)
    model = getattr(snn_module, cls)(config).to(device)
    if FORWARD_VERSION is not None:
        for m in model.modules():
            if isinstance(m, axonal_recdel):
                m.forward_version = FORWARD_VERSION
    optimizer, scheduler = init_optim_sche(model, config)

    train_loader, _valid, test_loader = load_dataset(config)

    # Fixed probe batch (same every epoch, and the same for every model): one utterance
    # set balanced over all 35 keywords, see balanced_probe_batch.
    p_in, p_tgt = balanced_probe_batch(test_loader)
    px = p_in.permute(1, 0, 2).float().to(device)
    py = p_tgt.to(device)

    # Fixed unit readout vectors for the Jacobian probe, identical across models/epochs.
    U = gf.unit_vector_probe(config.output_size, N_PROBES, PROBE_SEED, device)

    rows, rows_j = [], []
    for epoch in range(epochs):
        train_loader.reset()          # SSC_SpikeIterator is stateful (cf. ``experiments/train.py --dataset ssc``)
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)
        train(train_loader, model, optimizer, epoch, device, config)
        for sc in scheduler:
            sc.step()
        gf.set_probe_mode(model)
        row = gf.input_grad_over_time(model, px, py)   # (T,) final-step CE probe
        row_j = gf.jacobian_grad_over_time(model, px, U)   # (T,) scale-comparable probe
        rows.append(row)
        rows_j.append(row_j)
        r, rj = gf.effective_reach(row), gf.effective_reach(row_j)
        print(f"  {MODELS[mkey][1]:20s} epoch {epoch:3d}: "
              f"reach centroid={r['centroid']:.1f} steps  (tau={r['tau']:.1f}) | "
              f"jacobian centroid={rj['centroid']:.1f}  |J| sum={row_j.sum():.3e}")
    empty = np.zeros((0, config.time_window))
    return (np.stack(rows) if rows else empty,
            np.stack(rows_j) if rows_j else empty)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_heatmap(M, label, out_path, *, vmin, vmax):
    """Heatmap of g(t) over epochs. x = time-before-output, y = epoch, color = log norm.
    ``vmin``/``vmax`` are shared across models so the color scale is comparable."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    E, T = M.shape
    if E == 0:
        return
    M_tbo = M[:, ::-1]                                   # column k = time-before-output
    reach = np.array([gf.effective_reach(M[e])["centroid"] for e in range(E)])
    with plt.rc_context(PANEL_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
        im = ax.imshow(M_tbo, aspect="auto", origin="lower", cmap=style.SEQUENTIAL,
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       extent=[0, T - 1, 0, max(E - 1, 1)])
        ax.plot(reach, np.arange(E), color=style.ORANGE, lw=2.4, label="centroid reach")
        ax.set(title=label, xlabel="Time before output (steps)", ylabel="Training epoch")
        secax = ax.secondary_xaxis("top", functions=(lambda s: s * MS_PER_STEP,
                                                     lambda ms: ms / MS_PER_STEP))
        secax.set_xlabel("Time before output (ms)")
        ax.legend(loc="upper right", framealpha=0.9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                          label=r"$\|\partial L/\partial x_t\|$")
        cb.outline.set_linewidth(1.2)
        fig.tight_layout()
        _save_panel(fig, out_path)


def plot_reach_summary(mats, out_path):
    """Centroid reach vs epoch, both families overlaid."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=style.FIGSIZE_LINE)
    for mkey, M in mats.items():
        if M.shape[0] == 0:
            continue
        _cls, label, color, ls = MODELS[mkey]
        reach = np.array([gf.effective_reach(M[e])["centroid"] for e in range(M.shape[0])])
        ax.plot(np.arange(M.shape[0]), reach, color=color, ls=ls, lw=1.8, label=label)
    ax.set(title="Credit-assignment reach vs. training epoch",
           xlabel="Training epoch", ylabel="Gradient centroid reach (steps)")
    secax = ax.secondary_yaxis("right", functions=(lambda s: s * MS_PER_STEP,
                                                   lambda ms: ms / MS_PER_STEP))
    secax.set_ylabel("ms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_path)


# --------------------------------------------------------------------------- #
# Publication panel style (the heatmaps go into a paper figure as panels)
# --------------------------------------------------------------------------- #
# RC_LARGE_TEXT sizes + heavier axis chrome and full-strength ink on the ticks/labels,
# so the panel stays legible printed small. Arial is not installed here. Liberation
# Sans is metric-compatible (identical advance widths), so matplotlib lays the text out
# with it and _save_panel then rewrites the SVG's font-family to name Arial first. With
# svg.fonttype:none the text stays real text, so the paper renders in true Arial.
PANEL_RC = {
    **style.RC_LARGE_TEXT,
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.color": style.INK_PRIMARY,
    "ytick.color": style.INK_PRIMARY,
    "axes.labelcolor": style.INK_PRIMARY,
    "axes.edgecolor": style.INK_PRIMARY,
    # The secondary ms axis sits between the title and the plot, so the default pad
    # leaves the title looking like that axis's label.
    "axes.titlepad": 24.0,
}
FIGSIZE_PANEL = (5.8, 5.2)      # squarer than the old (6.2, 4.2) wide strip
_FONT_STACK = "Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"


def _save_panel(fig, path):
    """``save_fig`` + force the Arial-first font stack into the saved SVG (see PANEL_RC)."""
    out = save_fig(fig, path)
    if str(out).endswith(".svg"):
        svg = out.read_text()
        # Match the whole declaration up to ';' or the style attribute's closing '"',
        # it contains quoted family names, so the class must not stop at a quote.
        out.write_text(re.sub(r"font-family:[^;\"]*", f"font-family: {_FONT_STACK}", svg))
    return out


def _npz_path(out_dir, mkey):
    """One ``.npz`` per family: the families are trained by separate concurrent
    invocations (one per GPU), so a single shared file would race, whichever process
    saved last would drop the other's matrix, at 5.6 h a family."""
    local = os.path.join(out_dir, f"gradflow_matrices_{mkey}.npz")
    if os.path.exists(local):
        return local
    # fall back to the measurement recorded with the repository, so the figure can
    # be redrawn without retraining. A local run shadows it.
    shipped = paths.measurements("SSC", "gradflow_epochs", f"gradflow_matrices_{mkey}.npz")
    return shipped if os.path.exists(shipped) else local


def load_mats(out_dir, key="M"):
    """Every family's matrix written so far, so the plots span all of them.
    ``key``: ``"M"`` = cross-entropy probe, ``"J"`` = unit-vector Jacobian probe. Files
    written before the Jacobian probe existed carry only ``"M"``, and are skipped for ``"J"``."""
    mats = {}
    for mkey in MODELS:
        path = _npz_path(out_dir, mkey)
        if os.path.exists(path):
            data = np.load(path)
            if key in data:
                mats[mkey] = data[key]
    return mats


def plot_ratio_heatmap(M_a, M_b, label_a, label_b, out_path, *, normalize, clip=2.0):
    """Log-ratio map ``log10(g_a / g_b)`` over (epoch × time-before-output). A ratio, not a
    difference: g spans ~4 decades across lags, so a raw ``g_a, g_b`` is entirely dominated
    by the few timesteps nearest the readout and says nothing about the tail.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    E = min(M_a.shape[0], M_b.shape[0])
    A, B = M_a[:E, ::-1].astype(np.float64), M_b[:E, ::-1].astype(np.float64)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            A = A / A.sum(axis=1, keepdims=True)
            B = B / B.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.log10(A / B)
    R[~np.isfinite(R)] = np.nan            # dead epochs (g == 0) are undefined, not 0
    with plt.rc_context(PANEL_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
        im = ax.imshow(R, aspect="auto", origin="lower", cmap=style.DIVERGING,
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-clip, vmax=clip),
                       extent=[0, R.shape[1] - 1, 0, max(E - 1, 1)])
        kind = "shape" if normalize else "absolute"
        ax.set(title=f"Learned / fixed — {kind}",
               xlabel="Time before output (steps)", ylabel="Training epoch")
        secax = ax.secondary_xaxis("top", functions=(lambda s: s * MS_PER_STEP,
                                                     lambda ms: ms / MS_PER_STEP))
        secax.set_xlabel("Time before output (ms)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                          label=rf"$\log_{{10}}(g_{{\rm learned}}/g_{{\rm fixed}})$")
        cb.outline.set_linewidth(1.2)
        fig.tight_layout()
        _save_panel(fig, out_path)


def make_plots(mats, out_dir, decades=6.0, suffix=""):
    # Shared log color scale across both heatmaps so gradient magnitudes are comparable.
    # Clamp the range to `decades` below vmax: the vanished-gradient tail reaches numerical
    # noise, which would otherwise stretch LogNorm over ~20 decades and wash out the
    # between-model differences that live in the top ~3 decades.
    vmax = float(max((M.max() for M in mats.values() if M.size), default=1.0))
    vmin = vmax / (10.0 ** decades)
    for mkey, M in mats.items():
        plot_heatmap(M, MODELS[mkey][1],
                     os.path.join(out_dir, f"gradflow_heatmap_{mkey}{suffix}.svg"),
                     vmin=vmin, vmax=vmax)
    plot_reach_summary(mats, os.path.join(out_dir, f"gradflow_reach_vs_epoch{suffix}.svg"))
    # Learned-vs-fixed contrast, both ways: absolute (what the per-model heatmaps show)
    # and shape-only (what the reach centroid measures). For the Jacobian probe the
    # absolute map is the meaningful one. Its output vector is fixed by construction.
    if "ax_learned" in mats and "ax_fixed" in mats:
        for norm, tag in ((False, "abs"), (True, "shape")):
            plot_ratio_heatmap(
                mats["ax_learned"], mats["ax_fixed"], "Axonal learned", "Axonal fixed",
                os.path.join(out_dir, f"gradflow_ratio_{tag}_learned_vs_fixed{suffix}.svg"),
                normalize=norm)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Seeds to train + average over (default: [0]).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override config.epochs (e.g. 3 for a smoke test).")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS), help="Model families to run.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--jacobian", action="store_true",
                    help="Also draw the unit-vector Jacobian probe.")
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip training, replot from <out-dir>/gradflow_matrices.npz.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def plot_all():
        # The probe of the paper is the cross-entropy of the final-step readout
        # ("M"). The unit-vector Jacobian probe ("J") is also recorded in each
        # .npz, pass --jacobian to draw it as well.
        make_plots(load_mats(args.out_dir, "M"), args.out_dir)
        if getattr(args, "jacobian", False):
            jac = load_mats(args.out_dir, "J")
            if jac:
                make_plots(jac, args.out_dir, suffix="_jacobian")

    if args.plots_only:
        plot_all()
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mkey in args.models:
        per_seed, per_seed_j = [], []
        for seed in args.seeds:
            config = ssc.make_config()
            config.seed = seed
            if args.epochs is not None:
                config.epochs = args.epochs
            print(f"\n=== {MODELS[mkey][1]} | seed {seed} | {config.epochs} epochs ===")
            M, J = run_model(mkey, config, device, config.epochs)
            per_seed.append(M)
            per_seed_j.append(J)
        # average matrices over seeds (all share epoch count / T), then persist this
        # family immediately. A later family crashing must not cost the finished one.
        n_ep = min(m.shape[0] for m in per_seed)
        mean = lambda ms: np.stack([m[:n_ep] for m in ms], axis=0).mean(axis=0)
        np.savez(_npz_path(args.out_dir, mkey), M=mean(per_seed), J=mean(per_seed_j))
        print(f"Saved {mkey} matrices → {_npz_path(args.out_dir, mkey)}")

    # Plot every family on disk, not just this invocation's, so the shared log color
    # scale spans all of them once the last GPU finishes. Re-run with --plots-only to
    # refresh after a family lands later.
    plot_all()
    print(f"Figures (SVG) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
