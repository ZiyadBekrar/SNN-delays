"""The Mackey-Glass delay differential equation and its windowed dataset.

Integrated with forward Euler at dt = 0.1 and subsampled to unit intervals. The
first 5000 points are discarded as transient. Split contiguously in time (60/20/20)
so no timestep is shared across splits, then z-scored with the training statistics.
"""

import numpy as np
import torch

from torch.utils.data import Dataset


def generate_mackey_glass(total=6000, tau=30, dt=0.1, beta=0.2, gamma=0.1, n=10,
                          x0=1.2, discard=5000):
    """Euler-integrate the Mackey-Glass delay ODE and return `total` samples."""
    tau_steps = int(tau / dt)
    N = int((total + discard) / dt) + tau_steps + 1
    x = np.empty(N, dtype=np.float64)
    x[:tau_steps + 1] = x0

    for i in range(tau_steps, N - 1):
        x_tau = x[i - tau_steps]
        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[i]
        x[i + 1] = x[i] + dt * dx

    start = int(discard / dt) + tau_steps
    step = int(1.0 / dt)
    series = x[start::step]
    return series[:total].astype(np.float32)


class MackeyGlassDataset(Dataset):
    """Sliding windows over a 1-D series: input (window_size, 1), target the scalar
    `horizon` steps after the window's last sample."""

    def __init__(self, series, window_size, horizon):
        self.x = torch.from_numpy(series).float()
        self.W = int(window_size)
        self.H = int(horizon)

    def __len__(self):
        return len(self.x) - self.W - self.H + 1

    def __getitem__(self, idx):
        inp = self.x[idx: idx + self.W].unsqueeze(-1)  # (W, 1)
        tgt = self.x[idx + self.W + self.H - 1]        # scalar
        return inp, tgt
