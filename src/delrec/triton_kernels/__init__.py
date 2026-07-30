"""Triton kernels for spiking layers with learnable recurrent delays.

    utils                 host-side tile and sizing helpers
    surrogate             surrogate gradients and their name to id registry
    lif                   LIF charge / fire / reset step helpers
    delays                differentiable delay-kernel and tap builders (pure PyTorch)
    recdel_base           axonal persistent scan
    synaptic_eventdriven  synaptic spike-sparse scatter scan

Training code uses the autograd Functions and ``*_trainable_forward`` entry points.
On import this package points Triton at a driver-compatible ``ptxas`` when the host
driver predates Triton's bundled CUDA 12 toolchain, so the kernels also run on an A40
under the 470 driver.
"""

# Point Triton at a driver-compatible ptxas before importing any Triton kernel,
# so its CUDA-12 default doesn't produce cubin an older driver can't load
# (e.g. A40 on driver 470 / CUDA 11.4). No-op on CUDA-12+ drivers.
from ._driver_compat import configure_ptxas_for_driver as _configure_ptxas_for_driver
_configure_ptxas_for_driver()

# Primary entry points
from .recdel_base import (
    AxonalRecdelTriton,
    recdel_triton_forward,
    recdel_triton_backward,
    recdel_fwd_kernel,
    recdel_bwd_kernel,
)

# synaptic_recdel: per-synapse delays (recurrent_delays shape (N_in, N_out)).
# Fast path, exact vs the reference multi_step_forward_v2:
#
#  synaptic_eventdriven. The dense view is pessimistic: the per-synapse mask is
#    only ~1+2*sigma taps wide and spikes are sparse (~6-22% firing), so the true
#    work is O(B * n_fired * taps * N) << O(B * N^2 * L). A fused event-driven
#    scatter scan that compacts the fired sources each step realizes this: ~3-4x on
#    the forward and ~1.9-2.8x on the full fwd+bwd training step over v2 (firing-rate
#    dependent). The backward gradient flows densely (not spike-sparse), so it is a
#    dense reverse-BPTT scan + cuBLAS GEMMs.
#    Use via synaptic_recdel.forward_version = 'eventdriven'.
from .synaptic_eventdriven import (
    EventDrivenSyn,                       # autograd.Function (fwd scatter + bwd BPTT)
    eventdriven_trainable_forward,        # main entry: autograd-enabled (T,B,N) -> (T,B,N)
    eventdriven_scatter_compact_forward,  # forward-only spike-sparse scatter (no autograd)
)

# Building blocks (extension surface)
from .delays import build_mask, build_lag, build_lag_interp, build_syn_mask
from .surrogate import _surrogate_grad, _SURR_ID
from .lif import lif_fwd_step, lif_bwd_step

__all__ = [
    'AxonalRecdelTriton',
    'EventDrivenSyn', 'eventdriven_trainable_forward',
    'eventdriven_scatter_compact_forward',
    'recdel_triton_forward', 'recdel_triton_backward',
    'recdel_fwd_kernel', 'recdel_bwd_kernel',
    'build_mask', 'build_lag', 'build_lag_interp', 'build_syn_mask',
    'lif_fwd_step', 'lif_bwd_step',
]
