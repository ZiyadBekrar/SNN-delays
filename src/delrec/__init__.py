"""DelRec, learning delays in recurrent spiking neural networks.

Reference implementation for the paper of the same name. The delay machinery
lives in :mod:`delrec.delay_layers`. Everything else is the surrounding experiment
code.

    neurons    recurrent LIF layers with learnable axonal / synaptic delays
    kernels    fused Triton scans for those layers (exact vs. the torch reference)
    models     network zoos assembled from the layers, per benchmark
    data       dataset loaders
    training   per-benchmark train / test loops
"""

__version__ = "1.0.0"
