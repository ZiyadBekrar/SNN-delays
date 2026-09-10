"""Verify the hybrid delay networks on the target device before trusting a run.

  SNN_recurrent_hybrid   - the recurrent hybrid uses a fused Triton path on CUDA
                           (delrec.triton_kernels.synaptic_hybrid). Run one
                           forward + backward with forward_version='v2' (the
                           pure-torch reference) and with 'hybrid_triton', and
                           assert the output and every parameter gradient match.
                           "The Kaggle run didn't crash" only proves the kernels
                           compiled - this proves they compute the right thing.

  SNN_feedforward_hybrid - no recurrence and no delrec Triton kernel: its hybrid
                           delay rides DCLS's own CUDA conv. There is no second
                           path to diff against, so this is a smoke check - builds,
                           forward + backward run, output finite, the learned delay
                           leaf gets a finite gradient.

On CPU there is no Triton, so both fall back to the smoke check and the script
exits 0 (the equivalence check needs a CUDA GPU).

Run:  python experiments/check_hybrid_kernel.py [--device cuda]
Exit: 0 on pass (or CPU), 1 on mismatch.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from configs.perf_MEM import Config
from delrec import networks
from delrec.delay_layers import synaptic_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.utils import reset_states

MODELS = ["SNN_recurrent_hybrid", "SNN_feedforward_hybrid"]
T, B = 24, 8
FWD_ATOL, FWD_RTOL = 1e-4, 1e-4
GRAD_ATOL, GRAD_RTOL = 2e-4, 2e-3


def _config():
    cfg = Config()
    cfg.hidden_layers = [64]
    cfg.time_window = T
    cfg.batch_size = B
    cfg.use_batch_norm = False
    cfg.feedforward_dropout_rate = 0.0
    cfg.recurrent_dropout_rate = 0.0
    cfg.no_recurrence_in_last_layer = False  # keep a real recdel in the one hidden layer
    cfg.kernel_count = 1
    cfg.hybrid_max_synaptic_delay = 4
    return cfg


def _run(model, x, recdel_fv):
    for m in model.modules():
        if isinstance(m, synaptic_recdel):
            m.forward_version = recdel_fv
    model.zero_grad(set_to_none=True)
    reset_states(model)
    out = model(x)
    out.pow(2).mean().backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


def _has_triton_recdel(model):
    return any(isinstance(m, synaptic_recdel) and getattr(m, "_hybrid_delay", False)
              for m in model.modules())


def check(model_name, device):
    cfg = _config()
    torch.manual_seed(0)
    model = getattr(networks, model_name)(cfg).to(device)
    x = torch.randn(T, B, cfg.input_size, device=device,
                    generator=torch.Generator(device=device).manual_seed(7))

    y_ref, g_ref = _run(model, x, "v2")

    if not torch.isfinite(y_ref).all() or not all(torch.isfinite(g).all() for g in g_ref.values()):
        print(f"  {model_name}: FAIL - non-finite forward/gradient on the v2 path")
        return False

    if device.type != "cuda" or not _has_triton_recdel(model):
        # smoke check only: SNN_feedforward_hybrid always, everything on CPU.
        delay_leaves = [learned_delay_parameter(m, "P")
                        for m in model.modules() if isinstance(m, dcls_module)]
        delay_leaves += [learned_delay_parameter(m, "recurrent_delays")
                         for m in model.modules()
                         if isinstance(m, synaptic_recdel) and getattr(m, "_hybrid_delay", False)]
        if delay_leaves and not any(
                p.grad is not None and torch.isfinite(p.grad).all() for p in delay_leaves):
            print(f"  {model_name}: FAIL - learned delay leaf got no finite gradient")
            return False
        why = "CPU" if device.type != "cuda" else "no delrec Triton kernel on this path"
        print(f"  {model_name}: smoke OK ({why}; no v2-vs-fused comparison)")
        return True

    y_hyb, g_hyb = _run(model, x, "hybrid_triton")

    ok = True
    try:
        torch.testing.assert_close(y_hyb, y_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
    except AssertionError as exc:
        ok = False
        print(f"  {model_name}: FORWARD mismatch\n{exc}")
    if set(g_hyb) != set(g_ref):
        ok = False
        print(f"  {model_name}: grad key mismatch "
              f"{set(g_ref).symmetric_difference(g_hyb)}")
    for name in sorted(set(g_ref) & set(g_hyb)):
        try:
            torch.testing.assert_close(g_hyb[name], g_ref[name],
                                       atol=GRAD_ATOL, rtol=GRAD_RTOL)
        except AssertionError as exc:
            ok = False
            print(f"  {model_name}: GRAD mismatch at {name}\n{exc}")
    if ok:
        rel = max((g_hyb[n] - g_ref[n]).abs().max().item()
                  / (g_ref[n].abs().max().item() + 1e-12) for n in g_ref)
        print(f"  {model_name}: OK  (v2 vs hybrid_triton, max rel grad diff {rel:.2e})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"hybrid-kernel check on {device} (torch {torch.__version__})", flush=True)

    all_ok = all(check(name, device) for name in MODELS)
    print("RESULT:", "PASS" if all_ok else "FAIL", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
