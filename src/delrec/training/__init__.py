"""Per-benchmark training and evaluation loops.

Every module exposes ``train`` / ``test`` (``evaluate`` for the MG regression
task) and ``init_optim_sche``, which returns two optimizers and schedulers,
one for the weights, one for the delays, since they need different learning
rates.
"""
