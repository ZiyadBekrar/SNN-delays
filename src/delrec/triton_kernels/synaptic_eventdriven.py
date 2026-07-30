"""Fused spike-sparse kernel for per-connection delays.

The dense view of this recurrence is pessimistic: the per-connection delay kernel
is only a few taps wide and spikes are sparse, so the real work is far below the
O(B N^2 L) a dense scan would do. This compacts the neurons that fired at each
step and scatters only those, which is what realizes that saving.

The backward is a dense reverse-BPTT scan rather than a sparse one, deliberately:
the recurrence gradient is nonzero even for neurons that did not fire, because
the surrogate is nonzero near threshold. Exact against the PyTorch reference.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ed_rec_kernel(Y_ptr, Wt_ptr, dt_ptr, rec_ptr,
                   s, N, Lm1,
                   MAXR: tl.constexpr, BLOCK_I: tl.constexpr):
    """rec[b, i] = sum_o W[i,o] * sum_k clamp(s-|k-kc|,0)/s^2 * Y[b,o,k], kc = (Lm1), 1 -
    d[i,o] (position in the newest-last window Y of width Lm1).
    """
    b = tl.program_id(0)
    it = tl.program_id(1)
    i = it * BLOCK_I + tl.arange(0, BLOCK_I)          # receiver indices (cols)
    i_mask = i < N

    inv_s2 = 1.0 / (s * s)
    rec_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    y_base = b * N * Lm1                               # Y[b,0,0]

    for o in range(0, N):
        w = tl.load(Wt_ptr + o * N + i, mask=i_mask, other=0.0)        # (BLOCK_I,)
        d = tl.load(dt_ptr + o * N + i, mask=i_mask, other=0.0)        # (BLOCK_I,)
        kc = (Lm1 - 1.0) - d                                          # center in k-space
        kbase = tl.floor(kc)
        row = y_base + o * Lm1
        tapsum = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for dk in range(-MAXR, MAXR + 1):
            k = kbase + dk
            ki = k.to(tl.int32)
            wtap = tl.maximum(s - tl.abs(k - kc), 0.0) * inv_s2
            valid = (ki >= 0) & (ki <= Lm1 - 1) & i_mask
            yv = tl.load(Y_ptr + row + ki, mask=valid, other=0.0)
            tapsum += wtap * yv
        rec_acc += w * tapsum

    tl.store(rec_ptr + b * N + i, rec_acc, mask=i_mask)


def eventdriven_rec(Y, Wt, dt, s, MAXR, block_i=64):
    """One step: rec (B,N) from window Y (B,N,Lm1), pre-transposed Wt/dt (N,N)."""
    B, N, Lm1 = Y.shape
    rec = torch.empty(B, N, device=Y.device, dtype=Y.dtype)
    grid = (B, triton.cdiv(N, block_i))
    _ed_rec_kernel[grid](Y, Wt, dt, rec, float(s), N, Lm1, MAXR, BLOCK_I=block_i)
    return rec


@triton.jit
def _ed_scatter_fwd_kernel(x_ptr, Wt_ptr, dt_ptr, bias_ptr, drop_ptr, y_ptr, buf_ptr,
                           decay, v_th, s, T, B,
                           N: tl.constexpr, L: tl.constexpr, MAXR: tl.constexpr,
                           BLOCK_N: tl.constexpr, USE_BIAS: tl.constexpr):
    """Fused event-driven scatter scan for ONE batch sample (grid = (B,))."""
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK_N)
    im = i < N
    inv_s2 = 1.0 / (s * s)
    v = tl.zeros((BLOCK_N,), dtype=tl.float32)
    bias = tl.load(bias_ptr + i, mask=im, other=0.0) if USE_BIAS else 0.0
    dv = tl.load(drop_ptr + b * N + i, mask=im, other=1.0)            # (N,) fixed drop
    buf_b = buf_ptr + b * L * N
    ptr = 0
    for t in range(0, T):
        rec = tl.load(buf_b + ptr * N + i, mask=im, other=0.0)
        tl.store(buf_b + ptr * N + i, tl.zeros((BLOCK_N,), dtype=tl.float32), mask=im)
        xv = tl.load(x_ptr + t * B * N + b * N + i, mask=im, other=0.0)
        u = decay * v + xv + dv * (rec + bias)
        spike = (u >= v_th).to(tl.float32)
        v = u - spike * u                                            # hard reset to 0
        tl.store(y_ptr + t * B * N + b * N + i, spike, mask=im)
        tl.debug_barrier()                                          # spikes visible to all lanes
        ptr = (ptr + 1) % L
        for o in range(0, N):
            so = tl.load(y_ptr + t * B * N + b * N + o)             # scalar spike of source o
            wcol = tl.load(Wt_ptr + o * N + i, mask=im, other=0.0)  # W[i,o]
            dcol = tl.load(dt_ptr + o * N + i, mask=im, other=0.0)  # d[i,o]
            center = 1.0 + dcol
            lbase = tl.floor(center) - MAXR
            for dk in range(0, 2 * MAXR + 1):
                lf = lbase + dk
                li = lf.to(tl.int32)
                wtap = tl.maximum(s - tl.abs(lf - center), 0.0) * inv_s2 * so
                slot = (ptr + li - 1) % L
                valid = (li >= 1) & (li <= L - 1) & im & (wtap != 0.0)
                addr = buf_b + slot * N + i
                cur = tl.load(addr, mask=valid, other=0.0)
                tl.store(addr, cur + wcol * wtap, mask=valid)


@triton.jit
def _ed_scatter_compact_kernel(x_ptr, Wt_ptr, dt_ptr, s_ptr, bias_ptr, drop_ptr, y_ptr,
                               buf_ptr, fired_ptr, u_ptr,
                               decay, charge, v_th, v_reset_val, T, B,
                               N: tl.constexpr, L: tl.constexpr, MAXR: tl.constexpr,
                               BLOCK_N: tl.constexpr, USE_BIAS: tl.constexpr,
                               STORE_U: tl.constexpr, SOFT: tl.constexpr):
    """Step B: spike-sparse fused scatter."""
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK_N)
    im = i < N
    v = tl.zeros((BLOCK_N,), dtype=tl.float32)
    bias = tl.load(bias_ptr + i, mask=im, other=0.0) if USE_BIAS else 0.0
    dv = tl.load(drop_ptr + b * N + i, mask=im, other=1.0)
    buf_b = buf_ptr + b * L * N
    fired_b = fired_ptr + b * N
    ptr = 0
    for t in range(0, T):
        rec = tl.load(buf_b + ptr * N + i, mask=im, other=0.0)
        tl.store(buf_b + ptr * N + i, tl.zeros((BLOCK_N,), dtype=tl.float32), mask=im)
        xv = tl.load(x_ptr + t * B * N + b * N + i, mask=im, other=0.0)
        u = decay * v + charge + xv + dv * (rec + bias)
        spike = (u - v_th >= 0).to(tl.float32)
        if SOFT:
            v = u - spike * v_th
        else:
            v = u - spike * (u - v_reset_val)
        tl.store(y_ptr + t * B * N + b * N + i, spike, mask=im)
        if STORE_U:
            tl.store(u_ptr + t * B * N + b * N + i, u, mask=im)
        ptr = (ptr + 1) % L
        # compact fired sources: pos = inclusive prefix sum. Fired_b[pos-1] = index
        pos = tl.cumsum(spike, axis=0)
        nf = tl.sum(spike, axis=0).to(tl.int32)
        fmask = (spike > 0.0) & im
        # Cross-lane communication via global fired_b: __syncthreads orders the
        # writes but does NOT invalidate L1, so the readers below must bypass L1
        # (cache_modifier='.cg') or they race on stale cache. Barrier + .cg gives
        # the read-after-write coherence within the block.
        tl.store(fired_b + (pos.to(tl.int32) - 1), i.to(tl.int32), mask=fmask,
                 cache_modifier='.cg')
        tl.debug_barrier()
        for f in range(0, nf):
            o = tl.load(fired_b + f, volatile=True)
            wcol = tl.load(Wt_ptr + o * N + i, mask=im, other=0.0)
            dcol = tl.load(dt_ptr + o * N + i, mask=im, other=0.0)
            so = tl.load(s_ptr + o)                                   # per-source half-width
            inv_s2 = 1.0 / (so * so)
            center = 1.0 + dcol
            lbase = tl.floor(center) - MAXR
            for dk in range(0, 2 * MAXR + 1):
                lf = lbase + dk
                li = lf.to(tl.int32)
                wtap = tl.maximum(so - tl.abs(lf - center), 0.0) * inv_s2
                slot = (ptr + li - 1) % L
                valid = (li >= 1) & (li <= L - 1) & im & (wtap != 0.0)
                addr = buf_b + slot * N + i
                cur = tl.load(addr, mask=valid, other=0.0)
                tl.store(addr, cur + wcol * wtap, mask=valid)


def _syn_consts(layer):
    """LIF constants (decay_input=False), matching delrec.triton_kernels.axonal_eventdriven._ax_consts."""
    nm = layer.neuron_module
    vr = nm.v_reset
    soft = vr is None
    decay = 1.0 - 1.0 / float(nm.tau)
    charge = 0.0 if (vr is None or float(vr) == 0.0) else float(vr) / float(nm.tau)
    v_reset_val = 0.0 if vr is None else float(vr)
    return decay, charge, float(nm.v_threshold), v_reset_val, soft


def _syn_spread(layer, d, device, dtype):
    """Per-source (N_out) triangular half-width s[o] (N,) and its max. Under
    use_sig_p s = 1 + 2*sigma*sigmoid(p_spread). Else a uniform 1+sigma. Mirrors
    delays.build_syn_mask / synaptic_recdel._build_mask exactly (s indexed by N_out)."""
    N = d.shape[1]
    if layer.use_sig_p:
        s = 1.0 + 2.0 * float(layer.sigma) * torch.sigmoid(
            layer.p_spread.detach().to(device, dtype))
    else:
        s = torch.full((N,), 1.0 + float(layer.sigma), device=device, dtype=dtype)
    return s.contiguous(), float(s.max())


def eventdriven_scatter_compact_forward(layer, x_seq, num_warps=None, return_u=False):
    """Step B spike-sparse fused scatter forward. Exact vs v2 in the supported
    regime (decay_input=False. Use_sig_p and soft/hard reset both supported). With
    return_u=True also returns the pre-reset membrane u_seq (needed by backward)."""
    device, dtype = x_seq.device, torch.float32
    T, B, N = x_seq.shape
    x = x_seq.to(device, dtype).contiguous()

    W = layer.recurrent_weights.detach().to(device, dtype)
    d = layer.recurrent_delays.detach().to(device, dtype)
    if layer.round_delays:
        d = d.round()
    s_vec, s_max = _syn_spread(layer, d, device, dtype)
    L = int(math.ceil(1.0 + float(d.max()) + s_max)) + 1
    MAXR = int(math.ceil(s_max)) + 1
    Wt = W.t().contiguous()
    dt = d.t().contiguous()
    decay, charge, v_th, v_reset_val, soft = _syn_consts(layer)
    use_bias = layer.use_rec_bias
    bias = (layer.recurrent_bias.detach().to(device, dtype) if use_bias
            else torch.zeros(N, device=device, dtype=dtype))
    drop = torch.ones(B, N, device=device, dtype=dtype)

    buf = torch.zeros(B, L, N, device=device, dtype=dtype)
    y = torch.empty(T, B, N, device=device, dtype=dtype)
    fired = torch.empty(B, N, device=device, dtype=torch.int32)
    u_seq = (torch.empty(T, B, N, device=device, dtype=dtype) if return_u
             else torch.empty(1, device=device, dtype=dtype))
    BLOCK_N = 1 << (N - 1).bit_length()
    if num_warps is None:
        # Match threads to lanes: a block-wide cumsum compaction misbehaves when
        # num_warps*32 != BLOCK_N (idle lanes corrupt the prefix sum).
        num_warps = max(1, min(16, BLOCK_N // 32))
    _ed_scatter_compact_kernel[(B,)](
        x, Wt, dt, s_vec, bias, drop, y, buf, fired, u_seq,
        decay, charge, v_th, v_reset_val, T, B,
        N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, USE_BIAS=use_bias,
        STORE_U=return_u, SOFT=soft, num_warps=num_warps)
    return (y, u_seq) if return_u else y


def eventdriven_scatter_forward(layer, x_seq, num_warps=4):
    """Fused event-driven scatter forward (Step A). Forward-only, exact vs v2 in
    the supported regime: use_sig_p=False, decay_input=False, hard reset v_reset=0."""
    assert not layer.use_sig_p, "scatter kernel: use_sig_p not supported yet"
    nm = layer.neuron_module
    assert float(nm.v_reset) == 0.0, "scatter kernel: only hard reset v_reset=0"
    device, dtype = x_seq.device, torch.float32
    T, B, N = x_seq.shape
    x = x_seq.to(device, dtype).contiguous()

    W = layer.recurrent_weights.detach().to(device, dtype)
    d = layer.recurrent_delays.detach().to(device, dtype)
    if layer.round_delays:
        d = d.round()
    s = 1.0 + float(layer.sigma)
    L = int(math.ceil(1.0 + float(d.max()) + s)) + 1
    MAXR = int(math.ceil(s)) + 1
    Wt = W.t().contiguous()
    dt = d.t().contiguous()
    decay = 1.0 - 1.0 / float(nm.tau)
    v_th = float(nm.v_threshold)
    use_bias = layer.use_rec_bias
    bias = (layer.recurrent_bias.detach().to(device, dtype) if use_bias
            else torch.zeros(N, device=device, dtype=dtype))
    drop = torch.ones(B, N, device=device, dtype=dtype)

    buf = torch.zeros(B, L, N, device=device, dtype=dtype)
    y = torch.empty(T, B, N, device=device, dtype=dtype)
    BLOCK_N = 1 << (N - 1).bit_length()
    _ed_scatter_fwd_kernel[(B,)](
        x, Wt, dt, bias, drop, y, buf,
        decay, v_th, float(s), T, B,
        N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, USE_BIAS=use_bias,
        num_warps=num_warps)
    return y


def eventdriven_forward(layer, x_seq, block_i=64):
    """Full forward mirroring synaptic_recdel.multi_step_forward_v2, but with the
    per-step recurrent GEMM replaced by the gather kernel. Forward-only (no
    autograd through rec yet)."""
    assert not layer.use_sig_p, "event-driven kernel: use_sig_p not supported yet"
    device, dtype = x_seq.device, x_seq.dtype
    T, B, N = x_seq.shape

    W = layer.recurrent_weights.detach().to(device, dtype)            # (N_in, N_out)
    d = layer.recurrent_delays.detach().to(device, dtype)            # (N_in, N_out)
    if layer.round_delays:
        d = d.round()
    s = 1.0 + float(layer.sigma)
    L = int(math.ceil(1.0 + float(d.max()) + s)) + 1
    MAXR = int(math.ceil(s)) + 1

    Wt = W.t().contiguous()                                          # (o, i)
    dt = d.t().contiguous()                                          # (o, i)
    bias = layer.recurrent_bias.detach() if layer.use_rec_bias else None

    Y = torch.zeros(B, N, L - 1, device=device, dtype=dtype) if L >= 2 else None
    y_seq = []
    with torch.no_grad():
        for t in range(T):
            if L >= 2:
                rec = eventdriven_rec(Y, Wt, dt, s, MAXR, block_i)
            else:
                rec = torch.zeros(B, N, device=device, dtype=dtype)
            if bias is not None:
                rec = rec + bias
            rec = layer.dropout(rec)
            y = layer.neuron_module.single_step_forward(x_seq[t] + rec)
            if L >= 2:
                Y = torch.cat([Y[..., 1:], y.unsqueeze(-1)], dim=-1)
            y_seq.append(y)
    return torch.stack(y_seq)


# ======================================================================
# Backward (Milestone 2b): trainable end-to-end.
#
# The backward gradient flows DENSELY (every neuron's spike grad gathers from
# all receivers' future rec-grad), so unlike the forward it is not spike-sparse.
# Structure mirrors the axonal recdel_bwd_kernel:
#   - a fused reverse-BPTT kernel computes gr = dL/drec and dx = dL/dx via a
#     per-source future-gradient GATHER (the transpose of the forward scatter).
#   - dW, d_delays are then host-side: dW = sum_l mask[:,:,l] (.) C_l and
#     dd = sum_l dmask_dd[:,:,l] (.) C_l, with C_l[i,o] = sum_{t,b} gr[t,i] y[t-l,o]
#     a single cuBLAS GEMM per lag (no atomics).
# ======================================================================

from .lif import lif_bwd_step


@triton.jit
def _ed_bwd_kernel(gy_ptr, u_ptr, drop_ptr, W_ptr, d_ptr, s_ptr, gr_ptr, dx_ptr,
                   decay, v_th, v_reset_val, gamma, alpha, T, B,
                   N: tl.constexpr, L: tl.constexpr, MAXR: tl.constexpr,
                   BLOCK_N: tl.constexpr, SOFT: tl.constexpr, DETACH: tl.constexpr,
                   SURR: tl.constexpr):
    """Reverse BPTT for one sample (grid=(B,))."""
    b = tl.program_id(0)
    j = tl.arange(0, BLOCK_N)
    jm = j < N
    sval = tl.load(s_ptr + j, mask=jm, other=1.0)                    # per-source half-width
    inv_s2 = 1.0 / (sval * sval)
    lam_v = tl.zeros((BLOCK_N,), dtype=tl.float32)
    drp = tl.load(drop_ptr + b * N + j, mask=jm, other=1.0)
    for t in range(T - 1, -1, -1):
        gfut = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for i in range(0, N):
            w = tl.load(W_ptr + i * N + j, mask=jm, other=0.0)        # W[i,j] (coalesced over j)
            dij = tl.load(d_ptr + i * N + j, mask=jm, other=0.0)      # d[i,j]
            center = 1.0 + dij
            lbase = tl.floor(center) - MAXR
            for dk in range(0, 2 * MAXR + 1):
                lf = lbase + dk
                li = lf.to(tl.int32)
                tdst = t + li
                wtap = tl.maximum(sval - tl.abs(lf - center), 0.0) * inv_s2
                valid = (li >= 1) & (li <= L - 1) & (tdst < T) & jm & (wtap != 0.0)
                # Non-volatile is safe: gr[t+l] is written once (iteration t+l)
                # and only read at later iterations t<t+l, after that step's
                # debug_barrier -> no stale-cache hazard. Num_stages=1 keeps the
                # cross-iteration RAW unpipelined.
                grf = tl.load(gr_ptr + tdst * B * N + b * N + i, mask=valid, other=0.0)
                gfut += w * wtap * grf
        gy = tl.load(gy_ptr + t * B * N + b * N + j, mask=jm, other=0.0)
        u = tl.load(u_ptr + t * B * N + b * N + j, mask=jm, other=0.0)
        du = lif_bwd_step(gy + gfut, u, lam_v, v_th, v_reset_val, gamma, alpha,
                          SOFT, DETACH, SURR)
        gr_t = du * drp
        tl.store(gr_ptr + t * B * N + b * N + j, gr_t, mask=jm)
        tl.store(dx_ptr + t * B * N + b * N + j, du, mask=jm)
        tl.debug_barrier()                                           # gr[t] visible to earlier-t reads
        lam_v = decay * du


def _surrogate_ids(layer):
    """Match multi_step_forward_triton: (SURR id, gamma, alpha) from the config."""
    from delrec.utils import Triangle
    surr = layer.config.surrogate_function
    gamma = float(getattr(layer.config, 'surrogate_gamma', 1.0))
    if surr is Triangle.apply:
        return 0, gamma, 2.0
    if 'ATan' in type(surr).__name__:
        return 1, gamma, float(getattr(surr, 'alpha', 2.0))
    return 0, gamma, 2.0


class EventDrivenSyn(torch.autograd.Function):
    """Trainable event-driven synaptic_recdel (forward = spike-sparse scatter
    kernel, backward = dense reverse-BPTT kernel + host GEMMs). Supported regime:
    decay_input=False. Use_sig_p and soft/hard reset both supported. Dropout fixed
    mask. p_spread is a tensor input so its grad flows (zeros + no grad when off)."""

    @staticmethod
    def forward(ctx, x, W, d, p_spread, bias, drop, layer):
        device, dtype = x.device, torch.float32
        T, B, N = x.shape
        d_eff = d.detach().round() if layer.round_delays else d.detach()
        s_vec, s_max = _syn_spread(layer, d_eff, device, dtype)
        L = int(math.ceil(1.0 + float(d_eff.max()) + s_max)) + 1
        MAXR = int(math.ceil(s_max)) + 1

        Wt = W.detach().t().contiguous()
        dt = d_eff.t().contiguous()
        decay, charge, v_th, v_reset_val, soft = _syn_consts(layer)
        use_bias = bias is not None
        bias_k = bias.detach() if use_bias else torch.zeros(N, device=device, dtype=dtype)

        buf = torch.zeros(B, L, N, device=device, dtype=dtype)
        y = torch.empty(T, B, N, device=device, dtype=dtype)
        u = torch.empty(T, B, N, device=device, dtype=dtype)
        fired = torch.empty(B, N, device=device, dtype=torch.int32)
        BLOCK_N = 1 << (N - 1).bit_length()
        nw = max(1, min(16, BLOCK_N // 32))
        _ed_scatter_compact_kernel[(B,)](
            x.detach().contiguous(), Wt, dt, s_vec, bias_k, drop.detach(), y, buf, fired, u,
            decay, charge, v_th, v_reset_val, T, B,
            N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, USE_BIAS=use_bias,
            STORE_U=True, SOFT=soft, num_warps=nw)

        ctx.save_for_backward(W.detach(), d_eff, p_spread, s_vec, drop.detach(), u, y)
        ctx.layer = layer
        ctx.use_sig_p = layer.use_sig_p
        ctx.shapes = (T, B, N, L, MAXR, decay, charge, v_th, v_reset_val, soft, use_bias)
        return y

    @staticmethod
    def backward(ctx, gy):
        W, d_eff, p_spread, s_vec, drop, u, y = ctx.saved_tensors
        layer = ctx.layer
        T, B, N, L, MAXR, decay, charge, v_th, v_reset_val, SOFT, use_bias = ctx.shapes
        device, dtype = gy.device, torch.float32
        DETACH = bool(layer.config.detach_reset)
        SURR, gamma, alpha = _surrogate_ids(layer)

        gr = torch.zeros(T, B, N, device=device, dtype=dtype)
        dx = torch.empty(T, B, N, device=device, dtype=dtype)
        BLOCK_N = 1 << (N - 1).bit_length()
        nw = max(1, min(16, BLOCK_N // 32))      # 1 lane/thread is fastest here
        _ed_bwd_kernel[(B,)](
            gy.detach().contiguous(), u, drop, W.contiguous(), d_eff.contiguous(),
            s_vec, gr, dx,
            decay, v_th, v_reset_val, gamma, alpha, T, B,
            N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, SOFT=SOFT, DETACH=DETACH, SURR=SURR,
            num_warps=nw, num_stages=1)

        # dbias[i] = sum_{t,b} gr[t,b,i]
        dbias = gr.sum(dim=(0, 1)) if use_bias else None

        # C_l[i,o] = sum_{t>=l, b} gr[t,b,i] y[t-l,b,o]  (one cuBLAS GEMM per lag).
        # rec[i] = sum_o sum_l W[i,o] mask[i,o,l] y[t-l,o], so:
        #   dW[i,o]    = sum_l mask[i,o,l] C_l[i,o]
        #   dL/dmask[i,o,l] = W[i,o] C_l[i,o]  -> map to dd / dp_spread via autograd
        #     on build_syn_mask (exact, matches v2. Carries the W factor + use_sig_p).
        from .delays import build_syn_mask
        need_d = ctx.needs_input_grad[2]
        need_p = ctx.needs_input_grad[3] and ctx.use_sig_p
        d_ = d_eff.detach().requires_grad_(need_d)
        p_ = (p_spread.detach().requires_grad_(need_p) if ctx.use_sig_p else None)
        with torch.enable_grad():
            mask3, Lm = build_syn_mask(d_, layer.sigma, p_, ctx.use_sig_p)  # (N_in,N_out,Lm)
        Cstack = torch.zeros(N, N, Lm, device=device, dtype=dtype)          # (N_in,N_out,lag)
        for l in range(1, min(Lm, T)):
            M = T - l
            Cstack[:, :, l] = gr[l:T].reshape(M * B, N).t() @ y[0:M].reshape(M * B, N)
        dW = (mask3.detach() * Cstack).sum(-1)                              # (N_in,N_out)
        dmask_go = W.unsqueeze(-1) * Cstack                                 # dL/dmask
        dd = dp = None
        if need_d or need_p:
            grads = torch.autograd.grad(
                mask3, [t for t, n in ((d_, need_d), (p_, need_p)) if n],
                grad_outputs=dmask_go, allow_unused=True)
            gi = 0
            if need_d:
                dd = grads[gi]; gi += 1
            if need_p:
                dp = grads[gi]; gi += 1

        # signature: x, W, d, p_spread, bias, drop, layer
        return dx, dW, dd, dp, dbias, None, None


def eventdriven_trainable_forward(layer, x_seq):
    """Autograd-enabled event-driven forward for synaptic_recdel. Returns y with
    grads wired to recurrent_weights, recurrent_delays, recurrent_bias, p_spread, x.
    Dropout uses the layer's own RNG (fixed mask per forward, matching v2)."""
    T, B, N = x_seq.shape
    if layer.training and getattr(layer.config, 'recurrent_dropout_rate', 0.0) > 0:
        ones = torch.ones(B, N, device=x_seq.device, dtype=x_seq.dtype)
        drop = layer.dropout(ones)
    else:
        drop = torch.ones(B, N, device=x_seq.device, dtype=x_seq.dtype)
    bias = layer.recurrent_bias if layer.use_rec_bias else None
    p_spread = (layer.p_spread if layer.use_sig_p
                else torch.zeros(N, device=x_seq.device, dtype=x_seq.dtype))
    return EventDrivenSyn.apply(x_seq, layer.recurrent_weights, layer.recurrent_delays,
                                p_spread, bias, drop, layer)
