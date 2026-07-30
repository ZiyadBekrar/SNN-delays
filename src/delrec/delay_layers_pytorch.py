"""The delay layers, implemented in plain PyTorch.

    VanillaTorchScans    the delay-free recurrent baseline
    AxonalTorchScans     per-neuron delays:     v1, v2
    SynapticTorchScans   per-connection delays: v1, v2

``multi_step_forward_v2`` is the canonical version, a direct transcription of the
paper's scheduling equation as a dense sliding window over past spikes. It is the
reference every other implementation is checked against, and the CPU default.
``multi_step_forward_v1`` computes the same thing with the circular buffer of the
paper's algorithm listing.
"""

import torch



class VanillaTorchScans:
    """The delay-free recurrent scan of :class:`~delrec.delay_layers.vanilla_recurrent`."""

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype)   # (N_in, N_out)

        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            rec_now = torch.matmul(self.dropout(y_seq[-1]) if t > 0 else torch.zeros(B, N, device=device, dtype=dtype), W)  # (B, N_in)
    
            y = self.neuron_module.single_step_forward(x_seq[t] + rec_now)  # (B, N_out)
            
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)


class AxonalTorchScans:
    """Pure-PyTorch scans for per-neuron (axonal) delays."""

    def multi_step_forward_v1(self, x_seq: torch.Tensor):
        """Circular-buffer scatter, Algorithm 1 of the paper, read literally."""
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype).unsqueeze(0)   # (1, N_in, N_out)
        d = self.recurrent_delays.to(device=device, dtype=dtype)    # (N_out,)
        if self.round_delays:
            d = d + (d.round() - d).detach()

        if self.use_sig_p:
            s = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype) # (N_out,)
            s_max = s.max()
            s = s.unsqueeze(1) # (N_out, 1)
        else:
            s = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            s_max = s

        # Eq. (11): the kernel has finite support, so scheduling only ever needs
        # a window of L = ceil(1 + max_j d_j + (1 + sigma)) steps, not all of T.
        L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
        support = torch.arange(L, device=device, dtype=dtype)              # (L,)

        # Eq. (7): h_{sigma,d}(tau) = max(0, (1 + sigma (|tau) (1 + d)|) / (1 + sigma)^2),
        # the triangular spread centered on the delay. Under use_sig_p the half-width
        # is per-neuron (Eq. 14) rather than shared.
        mask = (torch.clamp(s - torch.abs(support[None, :] - (1.0 + d)[:, None]), min=0.0) / (s * s)).unsqueeze(0) # (1, N_out, L)

        buffer = torch.zeros(B, N, L, device=device, dtype=dtype)
        pointer = 0

        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            rec_now = buffer[:, :, pointer]                       # (B, N_in)
            if self.use_rec_bias:
                rec_now = rec_now + self.recurrent_bias

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N_out)

            y_masked = y.unsqueeze(2) * mask # (B, N_out, L)
            X_rec = torch.matmul(W, y_masked) # (B, N_in, L)
            
            buffer[:, :, pointer] = 0.0
            pointer = (pointer + 1) % L

            first_chunk = min(L - 1, L - pointer)
            if first_chunk > 0:
                buffer[:, :, pointer:pointer + first_chunk].add_(X_rec[:, :, 1:1 + first_chunk])
            remaining = L - 1 - first_chunk
            if remaining > 0:
                buffer[:, :, 0:remaining].add_(X_rec[:, :, 1 + first_chunk:1 + first_chunk + remaining])

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def multi_step_forward_v2(self, x_seq: torch.Tensor):
        """Dense sliding-window scan, the reference implementation."""
        
        device = x_seq.device
        dtype = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype).unsqueeze(0)  # (1, N, N)
    
        d = self.recurrent_delays.to(device=device, dtype=dtype)       # (N,)
        if self.round_delays:
            d = d + (d.round() - d).detach()

        if self.use_sig_p:
            s = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype)
            s_max = s.max()
            s = s.unsqueeze(1)  # (N, 1)
        else:
            s = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            s_max = s

        # Eq. (11): the kernel has finite support, so scheduling only ever needs
        # a window of L = ceil(1 + max_j d_j + (1 + sigma)) steps, not all of T.
        L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
        support = torch.arange(L, device=device, dtype=dtype)              # (L,)

        # Eq. (7): h_{sigma,d}(tau) = max(0, (1 + sigma (|tau) (1 + d)|) / (1 + sigma)^2),
        # the triangular spread centered on the delay. Under use_sig_p the half-width
        # is per-neuron (Eq. 14) rather than shared.
        mask = (torch.clamp(s - torch.abs(support[None, :] - (1.0 + d)[:, None]), min=0.0) / (s * s)).unsqueeze(0)
        # (1, N, L)

        rec_now_seq = []
        if self.store_v_seq:
            v_seq = []

        if L >= 2:
            # mask_flip[..., k] = mask[..., L-1-k] so it pairs with y_window[..., k] = y[t, (L-1-k)]
            mask_flip = mask[..., 1:L].flip(-1).contiguous()                # (1 or B, N, L-1)
            y_window = torch.zeros(B, N, L - 1, device=device, dtype=dtype)
        else:
            mask_flip = None
            y_window = None

        for t in range(T):
            if L >= 2:
                z = (mask_flip * y_window).sum(dim=-1)                     # (B, N)
                # Broadcast (1, N, N) or per-sample (B, N, N).
                rec_now = torch.matmul(W, z.unsqueeze(-1)).squeeze(-1)      # (B, N)
            else:
                rec_now = torch.zeros(B, N, device=device, dtype=dtype)

            if self.use_rec_bias:
                rec_now = rec_now + self.recurrent_bias

            rec_now_seq.append(rec_now)

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N)

            if L >= 2:
                y_window = torch.cat([y_window[..., 1:], y.unsqueeze(-1)], dim=-1)

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        self.rec_now_seq = torch.stack(rec_now_seq)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)


class SynapticTorchScans:
    """Pure-PyTorch scans for per-connection (synaptic) delays."""

    def _build_mask(self, d, device, dtype):
        """Per-synapse triangular delay kernel, shared by v1/v2.

        Returns (mask (N_in, N_out, L), L). Identical formula to the original
        circular-buffer forward."""
        N = self.neurons
        if self.use_sig_p:
            s = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype)  # (N,)
            s_max = s.max()
            s = s.view(1, N, 1)  # (1, N_out, 1)
        else:
            s = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            s_max = s

        L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
        support = torch.arange(L, device=device, dtype=dtype)  # (L,)
        mask = (torch.clamp(s - torch.abs(support[None, None, :] - (1.0 + d)[:, :, None]), min=0.0)
                / (s * s))  # (N_in, N_out, L)
        return mask, L

    def multi_step_forward_v1(self, x_seq: torch.Tensor):
        """Circular-buffer scatter (original implementation, kept as reference)."""
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype)   # (N_in, N_out)
        d = self.recurrent_delays.to(device=device, dtype=dtype)    # (N_in, N_out)
        if self.round_delays:
            d = d + (d.round() - d).detach()

        mask, L = self._build_mask(d, device, dtype)                # (N_in, N_out, L)

        buffer = torch.zeros(B, N, L, device=device, dtype=dtype) # (B, N_in, L)
        pointer = 0

        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            rec_now = buffer[:, :, pointer]                       # (B, N_in)
            if self.use_rec_bias:
                rec_now = rec_now + self.recurrent_bias

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N_out)

            X_rec = torch.einsum('bo,io,iol->bil', y, W, mask) # (B, N_in, L)

            buffer[:, :, pointer] = 0.0
            pointer = (pointer + 1) % L

            first_chunk = min(L - 1, L - pointer)
            if first_chunk > 0:
                buffer[:, :, pointer:pointer + first_chunk].add_(X_rec[:, :, 1:1 + first_chunk])
            remaining = L - 1 - first_chunk
            if remaining > 0:
                buffer[:, :, 0:remaining].add_(X_rec[:, :, 1 + first_chunk:1 + first_chunk + remaining])

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def multi_step_forward_v2(self, x_seq: torch.Tensor):
        """Sliding-window dense-GEMM equivalent of multi_step_forward_v1."""
        device = x_seq.device
        dtype = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype)   # (N_in, N_out)
        d = self.recurrent_delays.to(device=device, dtype=dtype)    # (N_in, N_out)
        if self.round_delays:
            d = d + (d.round() - d).detach()

        mask, L = self._build_mask(d, device, dtype)                # (N_in, N_out, L)

        # Wm[i,o,l] = W[i,o] * mask[i,o,l]. Pair Wm_flip[...,k] = Wm[..., L-1-k]
        # with yhist[...,k] = y[t-(L-1-k)] (same convention as axonal v2).
        if L >= 2:
            Wm = W.unsqueeze(-1) * mask                              # (N_in, N_out, L)
            Wm_flip = Wm[..., 1:L].flip(-1).reshape(N, N * (L - 1))  # (N_in, N_out*(L-1))
            y_window = torch.zeros(B, N, L - 1, device=device, dtype=dtype)
        else:
            Wm_flip = None
            y_window = None

        rec_now_seq = []
        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            if L >= 2:
                rec_now = y_window.reshape(B, N * (L - 1)) @ Wm_flip.t()   # (B, N_in)
            else:
                rec_now = torch.zeros(B, N, device=device, dtype=dtype)

            if self.use_rec_bias:
                rec_now = rec_now + self.recurrent_bias

            rec_now_seq.append(rec_now)

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N_out)

            if L >= 2:
                y_window = torch.cat([y_window[..., 1:], y.unsqueeze(-1)], dim=-1)

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        self.rec_now_seq = torch.stack(rec_now_seq)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)
