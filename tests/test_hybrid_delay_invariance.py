"""The hybrid delay classes must generalise the axonal models, not diverge from them.

``hybrid_offset_scale == 0`` (all per-synapse offsets zero) must reproduce the
paired axonal model byte-for-byte: same ``state_dict`` keys, same initial values at a
fixed seed, same forward. ``> 0`` must still build, keep the axonal delay-parameter
count (the offsets are frozen buffers), and run a finite forward.
"""

import math

import pytest
import torch

from configs.perf_MEM import Config
from delrec import networks
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.utils import reset_states


# (hybrid class, paired axonal model at maximum == 0)
PAIRS = [
    ("SNN_hybrid_feedforward_delays", "SNN_axonal_feedforward_delays"),
    ("SNN_recurrent_hybrid_delays", "SNN_axonal_recurrent_delays"),
]

T, B = 10, 4


def base_config(maximum):
    cfg = Config()
    cfg.input_size = 8
    cfg.output_size = 3
    cfg.hidden_layers = [16]
    cfg.time_window = T
    cfg.batch_size = B
    cfg.use_batch_norm = False
    cfg.feedforward_dropout_rate = 0.0
    cfg.recurrent_dropout_rate = 0.0
    cfg.no_delay_in_first_layer = False
    cfg.no_delay_in_last_layer = False
    cfg.no_recurrence_in_last_layer = False  # exercise a real recdel in the single hidden layer
    cfg.kernel_count = 1
    cfg.hybrid_offset_scale = maximum
    cfg.hybrid_delay_seed = 123
    return cfg


def _delay_leaves(model, name):
    kind = dcls_module if name == "P" else axonal_recdel
    return [learned_delay_parameter(m, name) for m in model.layers if isinstance(m, kind)]


def _probe(cfg):
    return torch.randn(cfg.time_window, B, cfg.input_size,
                       generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("hybrid_name, axonal_name", PAIRS)
def test_maximum_zero_is_the_axonal_model(hybrid_name, axonal_name):
    cfg = base_config(maximum=0)

    torch.manual_seed(0)
    hybrid = getattr(networks, hybrid_name)(cfg)
    torch.manual_seed(0)
    axonal = getattr(networks, axonal_name)(cfg)

    hs, as_ = hybrid.state_dict(), axonal.state_dict()
    assert set(hs) == set(as_), set(hs).symmetric_difference(as_)
    for key in as_:
        assert torch.equal(hs[key], as_[key]), f"init differs at {key}"

    x = _probe(cfg)
    hybrid.eval()
    axonal.eval()
    reset_states(hybrid)
    reset_states(axonal)
    torch.testing.assert_close(hybrid(x), axonal(x), atol=1e-6, rtol=0)


@pytest.mark.parametrize("hybrid_name, axonal_name", PAIRS)
def test_maximum_positive_keeps_axonal_delay_budget(hybrid_name, axonal_name):
    cfg = base_config(maximum=4)

    torch.manual_seed(0)
    hybrid = getattr(networks, hybrid_name)(cfg)
    torch.manual_seed(0)
    axonal = getattr(networks, axonal_name)(base_config(maximum=0))

    for name in ("P", "recurrent_delays"):
        h_leaves, a_leaves = _delay_leaves(hybrid, name), _delay_leaves(axonal, name)
        assert len(h_leaves) == len(a_leaves)
        for hl, al in zip(h_leaves, a_leaves):
            # one learned delay per source: same budget as the axonal model, even
            # though the feedforward leaf is laid out (1, 1, in, 1) not (1, in, 1, 1)
            assert hl.requires_grad
            assert hl.numel() == al.numel()

    offsets = [b for n, b in hybrid.named_buffers() if n.endswith(".offsets")]
    assert offsets, "maximum > 0 must store frozen per-synapse offsets as buffers"
    assert all(not b.requires_grad for b in offsets)
    assert all(int(b.max()) <= 4 and int(b.min()) >= 0 for b in offsets)

    x = _probe(cfg)
    hybrid.eval()
    reset_states(hybrid)
    out = hybrid(x)
    assert out.shape == (cfg.time_window, B, cfg.output_size)
    assert torch.isfinite(out).all()


def test_kernel_count_must_be_one():
    cfg = base_config(maximum=4)
    cfg.kernel_count = 2
    with pytest.raises(ValueError):
        networks.SNN_hybrid_feedforward_delays(cfg)


def test_maximum_must_be_nonnegative_integer():
    for bad in (-1, 1.5):
        cfg = base_config(maximum=bad)
        with pytest.raises(ValueError):
            networks.SNN_hybrid_feedforward_delays(cfg)


# ── dedicated hybrid Triton kernel (delrec.triton_kernels.synaptic_hybrid) ──────

def _triangular_syn_mask(d, sigma, L):
    """Per-synapse triangular delay kernel, matching triton_kernels.delays.build_syn_mask."""
    s = 1.0 + float(sigma)
    support = torch.arange(L, dtype=d.dtype).view(1, 1, L)
    return torch.clamp(s - (support - (1.0 + d).unsqueeze(-1)).abs(), min=0.0) / (s * s)


def _closed_form_delay_grad(Cstack, W, d, sigma):
    """Pure-torch mirror of _hybrid_dd_kernel: dd[i,o] = W[i,o] * sum_l C[i,o,l] *
    d/dd of the triangular kernel = sign(l-1-d)/s^2 inside its support."""
    s = 1.0 + float(sigma)
    lm = Cstack.shape[-1]
    a = torch.arange(lm, dtype=d.dtype).view(1, 1, lm) - 1.0 - d.unsqueeze(-1)
    dmask_dd = torch.sign(a) * (a.abs() < s).to(d.dtype) / (s * s)
    return (W.unsqueeze(-1) * Cstack * dmask_dd).sum(-1)


@pytest.mark.parametrize("sigma", [0.0, 0.5, 1.3])
def test_hybrid_delay_grad_closed_form_matches_autograd(sigma):
    """The kernel's closed-form triangular-kernel derivative must equal autograd
    through the soft mask (this is what EventDrivenSyn does today)."""
    torch.manual_seed(0)
    N, lm = 12, 6
    # non-integer d away from the |l-1-d| == s kinks so both paths use the same subgradient
    d = 0.3 + 3.0 * torch.rand(N, N)
    W = torch.randn(N, N)
    Cstack = torch.randn(N, N, lm)
    Cstack[:, :, 0] = 0.0  # lag 0 unused, matches the backward's range(1, Lm)

    d_ = d.clone().requires_grad_(True)
    mask = _triangular_syn_mask(d_, sigma, lm)
    (dd_ref,) = torch.autograd.grad(mask, d_, grad_outputs=W.unsqueeze(-1) * Cstack)

    torch.testing.assert_close(_closed_form_delay_grad(Cstack, W, d, sigma), dd_ref,
                               atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU + Triton")
def test_hybrid_triton_matches_v2_on_gpu():
    from delrec.delay_layers import synaptic_recdel

    cfg = base_config(maximum=4)
    torch.manual_seed(0)
    model = networks.SNN_recurrent_hybrid_delays(cfg).cuda()
    x = _probe(cfg).cuda()

    def run(fv):
        for m in model.modules():
            if isinstance(m, synaptic_recdel):
                m.forward_version = fv
        model.zero_grad(set_to_none=True)
        reset_states(model)
        out = model(x)
        out.pow(2).mean().backward()
        leaves = {n: p.grad.detach().clone()
                  for n, p in model.named_parameters() if p.grad is not None}
        return out.detach().clone(), leaves

    y_ref, g_ref = run("v2")
    y_hyb, g_hyb = run("hybrid_triton")

    torch.testing.assert_close(y_hyb, y_ref, atol=1e-4, rtol=1e-4)
    assert set(g_hyb) == set(g_ref)
    for name in g_ref:
        torch.testing.assert_close(g_hyb[name], g_ref[name], atol=2e-4, rtol=2e-3,
                                   msg=f"grad mismatch at {name}")
