"""Surrogate-gradient functions for spiking neurons, as @triton.jit helpers.

`_surrogate_grad` is called from the backward scan kernels with a `SURR`
constexpr selecting the family. New surrogates extend both `_SURR_ID` (the
host-side name→int map) and the `SURR ==` branch below.

The integer ids in `_SURR_ID` are the load-bearing contract: callers pass the
int through the autograd.Function into the kernel's `SURR: tl.constexpr`, and
`src/delrec/delay_layers_triton.py` hardcodes the same values
(0=triangle, 1=atan). Do not renumber them.
"""

import triton
import triton.language as tl


_SURR_ID = {'triangle': 0, 'atan': 1}


@triton.jit
def _surrogate_grad(x, gamma, alpha, SURR: tl.constexpr):
    if SURR == 0:   # Triangle: (1/gamma^2) * max(0, gamma - |x|)  
        inv = 1.0 / gamma
        return inv * inv * tl.maximum(gamma - tl.abs(x), 0.0)
    else:           # ATan(alpha): (alpha/2) / (1 + (pi/2 * alpha * x)^2)  
        z = 1.5707963267948966 * alpha * x   # (pi/2) * alpha * x
        return 0.5 * alpha / (1.0 + z * z)
