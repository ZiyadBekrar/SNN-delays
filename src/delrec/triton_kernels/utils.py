"""Host-side sizing helpers shared across the Triton SNN kernels.

Plain Python (no triton.jit): rounds neuron counts up to a power of two for the
(BLOCK_B, BLOCK_N) launch tiles.
"""


def _next_pow2(n):
    return 1 << (n - 1).bit_length()
