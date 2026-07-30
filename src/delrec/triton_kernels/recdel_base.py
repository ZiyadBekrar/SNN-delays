"""Persistent Triton scan for the axonal delay layer (shared mask and weights).

The PyTorch implementations issue ~6000 tiny ops per forward against ~10 us of actual
compute, and the recurrence is sequential over time, so the only way to remove that
overhead is to run the whole T-loop inside one kernel launch. That is
``recdel_fwd_kernel``, with ``recdel_bwd_kernel`` as the reverse BPTT scan.

Math per step, identical to ``multi_step_forward_v2``::

    z[j]   = sum_{l=1..L-1} mask[j,l] * y[t-l, j]   # per-neuron delayed gather
    r[i]   = sum_j W[i,j] * z[j]                    # recurrent matmul
    a[i]   = x[t,i] + drop[t,i] * r[i]
    u[i]   = decay*v[i] + charge_offset + a[i]      # LIF charge, decay_input=False
    s[i]   = (u[i] >= v_th)
    v[i]   = u[i] - s[i]*v_th                       # soft reset, or hard

The shared LIF dynamics are in ``lif.py``.
"""

import torch
import triton
import triton.language as tl

from .utils import _next_pow2
from .lif import lif_fwd_step, lif_bwd_step
from .surrogate import _surrogate_grad, _SURR_ID  # noqa: F401 (re-export convenience)
from .delays import build_mask, build_lag, build_lag_interp


# ──────────────────────────────────────────────────────────────────────────────
# Forward scan kernel (shared mask + shared weight)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def recdel_fwd_kernel(
    x_ptr, drop_ptr, wt_ptr, b_ptr, mask_ptr, lag_ptr, frac_ptr,
    y_ptr, u_ptr, rec_ptr, z_ptr,
    T, B, N,
    s_t, s_b, s_n,                 # strides of the (T,B,N) tensors (shared)
    swt_i, swt_o,                  # strides of Wt (N_in, N_out)
    sm_n, sm_l,                    # strides of mask (N, L)
    decay, charge_offset, v_th, v_reset_val,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr,
    L: tl.constexpr, SOFT: tl.constexpr, FWD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)        # (BB,)
    offs_n = tl.arange(0, BLOCK_N)                        # output neurons (N,)
    mb = offs_b < B
    mn = offs_n < N
    m_bn = mb[:, None] & mn[None, :]

    v = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for t in range(0, T):
        # Fused gather + recurrent matmul, tiled over the contraction dim (N_in)
        # so only a (BK, N) slab of Wt is resident at a time.
        #   r[b,i] = sum_j W[i,j] * z[b,j]
        # FWD==0 soft:   z[b,j] = sum_{l=1..L-1} mask[j,l] y[t-l,b,j]
        # FWD==1 round:  z[b,j] = y[t-lag[j], b, j]                     (single tap)
        # FWD==2 interp: z[b,j] = (1-frac[j]) y[t-lag[j],b,j] + frac[j] y[t-lag[j]-1,b,j]
        r = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, N, BK):
            offs_k = k0 + tl.arange(0, BK)               # source neurons (BK,)
            mk = offs_k < N
            if FWD == 0:
                z_blk = tl.zeros((BLOCK_B, BK), dtype=tl.float32)
                for l in range(1, L):
                    t_src = t - l
                    if t_src >= 0:
                        y_blk = tl.load(
                            y_ptr + t_src * s_t + offs_b[:, None] * s_b + offs_k[None, :] * s_n,
                            mask=mb[:, None] & mk[None, :], other=0.0)  # (BB, BK)
                        mcol = tl.load(mask_ptr + offs_k * sm_n + l * sm_l,
                                       mask=mk, other=0.0)              # (BK,)
                        z_blk += y_blk * mcol[None, :]
            elif FWD == 1:
                idx_k = tl.where(mk, offs_k, 0)                        # in-bounds index
                lag2 = tl.load(lag_ptr + idx_k[None, :])              # (1, BK), unmasked
                t_src = t - lag2                                       # (1, BK)
                vmask = mb[:, None] & (t_src >= 0) & mk[None, :]       # (BB, BK)
                z_blk = tl.load(
                    y_ptr + t_src * s_t + offs_b[:, None] * s_b + offs_k[None, :] * s_n,
                    mask=vmask, other=0.0)                             # (BB, BK)
            else:
                idx_k = tl.where(mk, offs_k, 0)                        # in-bounds index
                lag2 = tl.load(lag_ptr + idx_k[None, :])              # (1, BK) floor(1+d)
                fr = tl.load(frac_ptr + idx_k[None, :])              # (1, BK) fractional part
                t0 = t - lag2                                          # (1, BK) lower tap
                t1 = t0 - 1                                            # (1, BK) upper tap
                vm0 = mb[:, None] & (t0 >= 0) & mk[None, :]            # (BB, BK)
                vm1 = mb[:, None] & (t1 >= 0) & mk[None, :]            # (BB, BK)
                y0 = tl.load(
                    y_ptr + t0 * s_t + offs_b[:, None] * s_b + offs_k[None, :] * s_n,
                    mask=vm0, other=0.0)                               # (BB, BK)
                y1 = tl.load(
                    y_ptr + t1 * s_t + offs_b[:, None] * s_b + offs_k[None, :] * s_n,
                    mask=vm1, other=0.0)                               # (BB, BK)
                z_blk = (1.0 - fr) * y0 + fr * y1                      # (BB, BK)
            wt_blk = tl.load(
                wt_ptr + offs_k[:, None] * swt_i + offs_n[None, :] * swt_o,
                mask=mk[:, None] & mn[None, :], other=0.0)             # (BK, N)
            r += tl.dot(z_blk, wt_blk, input_precision="ieee")         # (BB, N)
            # stash this neuron-block of z for the backward dW reduction
            tl.store(z_ptr + t * s_t + offs_b[:, None] * s_b + offs_k[None, :] * s_n,
                     z_blk, mask=mb[:, None] & mk[None, :])

        # recurrent bias (per output neuron i), added to r before the dropout/charge:
        #   r[b,i] = sum_j W[i,j] z[b,j] + b[i]
        bvec = tl.load(b_ptr + offs_n, mask=mn, other=0.0)             # (BLOCK_N,)
        r += bvec[None, :]

        xv = tl.load(x_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                     mask=m_bn, other=0.0)
        dv = tl.load(drop_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                     mask=m_bn, other=0.0)

        a = xv + dv * r
        u, spike, v = lif_fwd_step(v, a, decay, charge_offset, v_th, v_reset_val, SOFT)

        tl.store(y_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                 spike, mask=m_bn)
        tl.store(u_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                 u, mask=m_bn)
        tl.store(rec_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                 r, mask=m_bn)
        # y[t] is read as y[t-l] next iterations from a different tile layout
        # (BB,BK gather vs BB,BLOCK_N store) -> fence so the write is visible.
        tl.debug_barrier()


# ──────────────────────────────────────────────────────────────────────────────
# Host wrappers
# ──────────────────────────────────────────────────────────────────────────────

def recdel_triton_forward(x, W, mask, drop, decay, charge_offset, v_th,
                          v_reset, lag=None, frac=None, approx=False, fwd_mode=None,
                          block_b=16, bk=64, bias=None):
    """Run the forward scan kernel."""
    assert x.is_cuda, f"x must be on CUDA, got device={x.device}"
    assert x.dtype == torch.float32, f"x must be float32, got dtype={x.dtype}"
    T, B, N = x.shape
    L = mask.shape[1]
    if fwd_mode is None:
        fwd_mode = 1 if approx else 0
    x = x.contiguous()
    drop = drop.contiguous()
    Wt = W.t().contiguous()                                           # (N_in, N_out)
    mask = mask.contiguous()
    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=x.dtype)
    bias = bias.contiguous()
    if lag is None:
        lag = torch.ones(N, device=x.device, dtype=torch.int32)
    lag = lag.contiguous()
    if frac is None:
        frac = torch.zeros(N, device=x.device, dtype=x.dtype)
    frac = frac.contiguous()

    y = torch.empty_like(x)
    u = torch.empty_like(x)
    rec = torch.empty_like(x)
    z = torch.empty_like(x)

    BLOCK_N = _next_pow2(N)
    bk = min(bk, BLOCK_N)
    soft = (v_reset is None)
    v_reset_val = 0.0 if v_reset is None else float(v_reset)

    grid = (triton.cdiv(B, block_b),)
    recdel_fwd_kernel[grid](
        x, drop, Wt, bias, mask, lag, frac,
        y, u, rec, z,
        T, B, N,
        x.stride(0), x.stride(1), x.stride(2),
        Wt.stride(0), Wt.stride(1),
        mask.stride(0), mask.stride(1),
        float(decay), float(charge_offset), float(v_th), float(v_reset_val),
        BLOCK_B=block_b, BLOCK_N=BLOCK_N, BK=bk, L=L, SOFT=soft, FWD=fwd_mode,
        num_stages=1,   # the y[t-l] read-after-write across iters must not pipeline
    )
    return y, u, rec, z


# ──────────────────────────────────────────────────────────────────────────────
# Backward (BPTT) scan kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def recdel_bwd_kernel(
    gy_ptr, u_ptr, drop_ptr, w_ptr, mask_ptr, lag_ptr, y_ptr,
    gz_ptr, gr_ptr, dx_ptr, dd_ptr,   # outputs (T,B,N). gz doubles as future-grad history
    T, B, N,
    s_t, s_b, s_n,
    sw_i, sw_o,                     # strides of W (N_out=i, N_in=j): W[i,j] at i*sw_i + j*sw_o
    sm_n, sm_l,                     # strides of mask (N, L)
    decay, v_th, v_reset_val, gamma, alpha,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr,
    L: tl.constexpr, SOFT: tl.constexpr, DETACH: tl.constexpr, SURR: tl.constexpr,
    APPROX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)                       # neuron index (i for W rows)
    mb = offs_b < B
    mn = offs_n < N
    m_bn = mb[:, None] & mn[None, :]

    lam_v = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)   # carry dL/dv[t]
    dd_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)          # straight-through delay grad (approx)
    idx_n = tl.where(mn, offs_n, 0)
    lag2 = tl.load(lag_ptr + idx_n[None, :])             # (1, N) int, unmasked (in-bounds)

    for t in range(T - 1, -1, -1):
        # gfuture[i] = d L / d y[t,i] contributed via the recurrence.
        # exact:  sum_{l=1..L-1} mask[i,l] gz[t+l, i]
        # approx: gz[t+lag[i], i]                              (single tap)
        gfut = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        if APPROX:
            t_dst = t + lag2                                  # (1, N)
            vmask = mb[:, None] & (t_dst < T) & mn[None, :]   # (BB, N)
            gfut = tl.load(
                gz_ptr + t_dst * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                mask=vmask, other=0.0)
        else:
            for l in range(1, L):
                t_dst = t + l
                if t_dst < T:
                    gzf = tl.load(
                        gz_ptr + t_dst * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                        mask=m_bn, other=0.0)
                    mcol = tl.load(mask_ptr + offs_n * sm_n + l * sm_l, mask=mn, other=0.0)
                    gfut += gzf * mcol[None, :]

        gy = tl.load(gy_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                     mask=m_bn, other=0.0)
        u = tl.load(u_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                    mask=m_bn, other=0.0)
        drp = tl.load(drop_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                      mask=m_bn, other=0.0)

        gY = gy + gfut
        du = lif_bwd_step(gY, u, lam_v, v_th, v_reset_val, gamma, alpha, SOFT, DETACH, SURR)

        gr = du * drp

        # gz[b,j] = sum_i gr[b,i] * W[i,j]   -> tile over output j-blocks (W columns)
        for j0 in range(0, N, BK):
            offs_j = j0 + tl.arange(0, BK)
            mj = offs_j < N
            w_blk = tl.load(
                w_ptr + offs_n[:, None] * sw_i + offs_j[None, :] * sw_o,
                mask=mn[:, None] & mj[None, :], other=0.0)          # (N_i, BK_j)
            gz_blk = tl.dot(gr, w_blk, input_precision="ieee")      # (BB, BK_j)
            tl.store(gz_ptr + t * s_t + offs_b[:, None] * s_b + offs_j[None, :] * s_n,
                     gz_blk, mask=mb[:, None] & mj[None, :])
        # gz[t] is read as gz[t+l] (next iters) and reloaded full below, from a
        # different tile layout than the (BB,BK) store -> fence for visibility.
        tl.debug_barrier()

        tl.store(gr_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                 gr, mask=m_bn)
        tl.store(dx_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                 du, mask=m_bn)

        if APPROX:
            # straight-through delay gradient:
            #   dz/dd ~ y[t-lag-1], y[t-lag]   (tap slope), summed over t,b
            gz_full = tl.load(
                gz_ptr + t * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                mask=m_bn, other=0.0)
            t0 = t - lag2                                     # (1, N)
            t1 = t0 - 1
            y0 = tl.load(
                y_ptr + t0 * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                mask=mb[:, None] & (t0 >= 0) & mn[None, :], other=0.0)
            y1 = tl.load(
                y_ptr + t1 * s_t + offs_b[:, None] * s_b + offs_n[None, :] * s_n,
                mask=mb[:, None] & (t1 >= 0) & mn[None, :], other=0.0)
            dd_acc += tl.sum(gz_full * (y1 - y0), axis=0)

        lam_v = decay * du

    if APPROX:
        tl.atomic_add(dd_ptr + offs_n, dd_acc, mask=mn)


def recdel_triton_backward(gy, u, drop, W, mask, z_seq, y_seq,
                           decay, v_th, v_reset, detach_reset, surr_id,
                           gamma=1.0, alpha=2.0, lag=None, approx=False, block_b=16, bk=64):
    """Reverse BPTT scan + host-side dW / db reduction."""
    T, B, N = gy.shape
    L = mask.shape[1]
    gy = gy.contiguous()
    u = u.contiguous()
    drop = drop.contiguous()
    W = W.contiguous()
    mask = mask.contiguous()
    if lag is None:
        lag = torch.ones(N, device=gy.device, dtype=torch.int32)
    lag = lag.contiguous()

    gz = torch.empty_like(gy)
    gr = torch.empty_like(gy)
    dx = torch.empty_like(gy)
    dd = torch.zeros(N, device=gy.device, dtype=gy.dtype)

    BLOCK_N = _next_pow2(N)
    bk = min(bk, BLOCK_N)
    soft = (v_reset is None)
    v_reset_val = 0.0 if v_reset is None else float(v_reset)

    grid = (triton.cdiv(B, block_b),)
    recdel_bwd_kernel[grid](
        gy, u, drop, W, mask, lag, y_seq,
        gz, gr, dx, dd,
        T, B, N,
        gy.stride(0), gy.stride(1), gy.stride(2),
        W.stride(0), W.stride(1),
        mask.stride(0), mask.stride(1),
        float(decay), float(v_th), float(v_reset_val), float(gamma), float(alpha),
        BLOCK_B=block_b, BLOCK_N=BLOCK_N, BK=bk, L=L,
        SOFT=soft, DETACH=bool(detach_reset), SURR=surr_id, APPROX=approx,
        num_stages=1,   # the gz[t+l] read-after-write across iters must not pipeline
    )

    # dW[i,j] = sum_{t,b} gr[t,b,i] * z[t,b,j]
    gr2 = gr.reshape(T * B, N)
    z2 = z_seq.reshape(T * B, N)
    dW = gr2.t() @ z2                                         # (N_out=i, N_in=j)
    # recurrent bias grad: db[i] = sum_{t,b} dL/dr[t,b,i] = sum_{t,b} gr[t,b,i]
    db = gr2.sum(0)                                           # (N,)

    if approx:
        return dx, dW, db, None, dd

    # dmask[j,l] = sum_{t,b} gz[t,b,j] * y[t-l,b,j]  (only lags l<T have valid pairs)
    dmask = torch.zeros(N, L, device=gy.device, dtype=gy.dtype)
    for l in range(1, min(L, T)):
        dmask[:, l] = (gz[l:T] * y_seq[0:T - l]).sum(dim=(0, 1))

    return dx, dW, db, dmask, None


# ──────────────────────────────────────────────────────────────────────────────
# autograd.Function wrapper (shared-mask / shared-weight)
# ──────────────────────────────────────────────────────────────────────────────

class AxonalRecdelTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, bias, d, p_spread, drop,
                sigma, tau, v_th, v_reset, use_sig_p, detach_reset, surr_id,
                gamma=1.0, approx=False, ste_mode=0, alpha=2.0):
        # ste_mode: 0 -> normal (soft/exact or straight-through approx, per `approx`),
        #           1 -> STE rounded forward, 2 -> STE 2-tap interp forward.
        #           STE modes use a hard-tap forward but the fully-soft backward.
        decay = 1.0 - 1.0 / tau
        charge_offset = 0.0 if (v_reset is None or v_reset == 0.0) else float(v_reset) / tau
        N = x.shape[2]
        frac = None
        if ste_mode:
            # Build the SOFT mask (full L) for the backward, plus the hard taps for
            # the forward. Backward then runs the unchanged soft (exact) path.
            with torch.no_grad():
                mask, L = build_mask(d, sigma, p_spread, use_sig_p)
                if ste_mode == 1:
                    lag = build_lag(d, L)
                    fwd_mode = 1
                else:
                    lag, frac = build_lag_interp(d, L)
                    fwd_mode = 2
            approx = False   # backward must be the soft path
        elif approx:
            with torch.no_grad():
                Lmax = int(torch.round(d).max().item()) + 2
                lag = build_lag(d, Lmax)
            mask = torch.zeros(N, 2, device=x.device, dtype=x.dtype)   # dummy (unused)
            fwd_mode = 1
        else:
            with torch.no_grad():
                mask, _ = build_mask(d, sigma, p_spread, use_sig_p)
            lag = torch.ones(N, device=x.device, dtype=torch.int32)
            fwd_mode = 0

        y, u, rec, z = recdel_triton_forward(
            x, W, mask, drop, decay, charge_offset, v_th, v_reset,
            lag=lag, frac=frac, fwd_mode=fwd_mode, bias=bias,
        )
        ctx.save_for_backward(u, drop, W, mask, z, y, d, p_spread, lag)
        ctx.decay = decay
        ctx.v_th = v_th
        ctx.v_reset = v_reset
        ctx.use_sig_p = use_sig_p
        ctx.detach_reset = detach_reset
        ctx.surr_id = surr_id
        ctx.sigma = sigma
        ctx.gamma = gamma
        ctx.alpha = alpha
        ctx.approx = approx
        return y

    @staticmethod
    def backward(ctx, gy):
        u, drop, W, mask, z, y, d, p_spread, lag = ctx.saved_tensors
        dx, dW, db, dmask, dd_st = recdel_triton_backward(
            gy.contiguous(), u, drop, W, mask, z, y,
            ctx.decay, ctx.v_th, ctx.v_reset, ctx.detach_reset, ctx.surr_id,
            gamma=ctx.gamma, alpha=ctx.alpha, lag=lag, approx=ctx.approx,
        )
        dd = dp = None
        # arg order: x=0, W=1, bias=2, d=3, p_spread=4, ...
        db = db if ctx.needs_input_grad[2] else None
        need_d = ctx.needs_input_grad[3]
        need_p = ctx.needs_input_grad[4] and ctx.use_sig_p

        if ctx.approx:
            # straight-through delay grad comes straight from the kernel. No spread.
            if need_d:
                dd = dd_st
        else:
            # dmask -> dd, dp_spread through the (cheap) PyTorch mask construction
            if need_d or need_p:
                d_ = d.detach().requires_grad_(need_d)
                p_ = (p_spread.detach().requires_grad_(need_p)
                      if ctx.use_sig_p else None)
                with torch.enable_grad():
                    mask2, _ = build_mask(d_, ctx.sigma, p_, ctx.use_sig_p)
                grads = torch.autograd.grad(
                    mask2, [t for t, n in ((d_, need_d), (p_, need_p)) if n],
                    grad_outputs=dmask, allow_unused=True)
                gi = 0
                if need_d:
                    dd = grads[gi]; gi += 1
                if need_p:
                    dp = grads[gi]; gi += 1

        # signature: x, W, bias, d, p_spread, drop, sigma, tau, v_th, v_reset,
        #            use_sig_p, detach_reset, surr_id, gamma, approx, ste_mode, alpha
        return (dx, dW, db, dd, dp, None, None, None, None, None, None, None, None, None, None, None, None)
