"""The same delay layers as fused GPU kernels, an optional accelerator.

    AxonalTritonScans    per-neuron delays:     'triton_exact'
    SynapticTritonScans  per-connection delays: 'eventdriven'
    SynapticHybridScans  hybrid delays:         'hybrid_triton'

Nothing here changes what the layers compute: spikes are bit-identical to
:mod:`delrec.delay_layers_pytorch` and gradients equal to floating-point tolerance,
which ``tests/test_kernel_equivalence.py`` asserts. Only speed differs, these keep the
whole time loop inside one kernel launch. Without Triton or a GPU the layers fall back
to PyTorch and everything still works. The raw kernels live in
:mod:`delrec.triton_kernels`.
"""

import torch


class AxonalTritonScans:
    """Fused scans for per-neuron (axonal) delays."""

    def multi_step_forward_triton(self, x_seq: torch.Tensor, approx: bool = False,
                                  ste_mode: int = 0):
        """Persistent Triton scan (see the delrec.triton_kernels package)."""
        from delrec.triton_kernels import AxonalRecdelTriton
        from delrec.utils import Triangle

        T, B, N = x_seq.shape
        # surrogate id + ATan slope, matching config.surrogate_function exactly.
        surr = self.config.surrogate_function
        if surr is Triangle.apply:
            sid, alpha = 0, 2.0
        elif 'ATan' in type(surr).__name__:
            sid, alpha = 1, float(getattr(surr, 'alpha', 2.0))
        else:
            sid, alpha = 0, 2.0

        # Recurrent dropout. SpikingJelly's layer.Dropout (used by the v1/v2 path)
        # samples ONE mask per forward and holds it FIXED across all T timesteps
        # (re-sampling per step would average the dropout out and change the
        # regularization). We reuse the layer's own self.dropout to generate
        # that mask, so it matches both the fixed-mask semantics AND the exact
        # RNG draw of the reference path -> identical training trajectory.
        if self.training and self.config.recurrent_dropout_rate > 0:
            ones = torch.ones(B, N, device=x_seq.device, dtype=x_seq.dtype)
            keep_mask = self.dropout(ones)                           # (B, N), 0 or 1/(1-p)
            drop = keep_mask.unsqueeze(0).expand(T, B, N).contiguous()  # same mask every step
        else:
            drop = torch.ones(T, B, N, device=x_seq.device, dtype=x_seq.dtype)

        p_spread = (self.p_spread if self.use_sig_p
                    else torch.zeros(N, device=x_seq.device, dtype=x_seq.dtype))
        bias = (self.recurrent_bias if self.use_rec_bias
                else torch.zeros(N, device=x_seq.device, dtype=x_seq.dtype))

        # Triangle surrogate width. The kernel's backward must use the SAME gamma as
        # the reference v2 path (delrec.utils.Triangle, called by config.surrogate_function).
        # Default 1.0 matches Triangle.apply's own default when no gamma is configured.
        gamma = float(getattr(self.config, 'surrogate_gamma', 1.0))

        if ste_mode:
            ste_mode = 1 if getattr(self.config, 'ste_round_forward', True) else 2

        # round_delays parity with v1/v2: feed STE-rounded delays (forward uses
        # round(d), backward grad flows straight through to recurrent_delays).
        d_in = self.recurrent_delays
        if self.round_delays:
            d_in = d_in + (d_in.round() - d_in).detach()

        y = AxonalRecdelTriton.apply(
            x_seq, self.recurrent_weights, bias, d_in, p_spread, drop,
            float(self.sigma), self.neuron_module.tau, self.neuron_module.v_threshold,
            self.neuron_module.v_reset, self.use_sig_p, self.config.detach_reset,
            sid, gamma, approx, ste_mode, alpha)

        self.x_seq = x_seq
        self.rec_now_seq = None
        if self.store_v_seq:
            self.v_seq = None
        return y


class SynapticTritonScans:
    """Fused spike-sparse scan for per-connection (synaptic) delays."""

    def multi_step_forward_eventdriven(self, x_seq: torch.Tensor):
        """Spike-sparse fused event-driven scatter scan (delrec.triton_kernels.synaptic_eventdriven).
        Exact vs v2 (fwd bit-identical, grads match): supports use_sig_p and soft/hard
        reset. Requires decay_input=False. ~2.4-5x over v2 on a HAR fwd+bwd step
        (firing-rate dependent)."""
        from delrec.triton_kernels.synaptic_eventdriven import eventdriven_trainable_forward
        y = eventdriven_trainable_forward(self, x_seq)
        self.x_seq = x_seq
        self.rec_now_seq = None
        if self.store_v_seq:
            self.v_seq = None
        return y


class SynapticHybridScans:
    """Dedicated fused scan for hybrid recurrent delays.

    A hybrid ``synaptic_recdel`` (learned per-source axonal base + frozen integer
    per-synapse offsets) is numerically an ordinary per-synapse delay, so the
    forward/recurrence reuse the event-driven kernels; only the delay gradient
    gets a fused closed-form reduction. See ``delrec.triton_kernels.synaptic_hybrid``.
    """

    def multi_step_forward_hybrid_triton(self, x_seq: torch.Tensor):
        """Autograd-enabled hybrid-delay forward. Raises on an unsupported regime
        or any Triton failure so the caller can fall back to 'eventdriven' / 'v2'."""
        from delrec.triton_kernels.synaptic_hybrid import hybrid_trainable_forward
        return hybrid_trainable_forward(self, x_seq)

