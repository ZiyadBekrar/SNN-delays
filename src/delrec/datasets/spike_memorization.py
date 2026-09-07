"""Fixed random patterns and independent random labels for memorization."""

import torch
from torch.utils.data import Dataset


class SpikeMemorization(Dataset):
    """Return (x, y), with x shaped (time, input neurons).

    Temporal patterns have one spike per input neuron; spatial patterns have
    constant random currents. Samples and labels are generated only once.
    """

    def __init__(self, num_samples: int, n_input: int, seq_length: int,
                 n_classes: int, seed: int = 0, input_gain: float = 1.0,
                 task_type: str = "temporal"):
        self.num_samples = int(num_samples)
        self.n_input = int(n_input)
        self.seq_length = int(seq_length)
        self.n_classes = int(n_classes)
        self.input_gain = float(input_gain)
        self.task_type = task_type
        if min(self.num_samples, self.n_input, self.seq_length, self.n_classes) < 1:
            raise ValueError("Sample, input, time and class counts must be positive.")

        g = torch.Generator().manual_seed(int(seed))
        if task_type == "temporal":
            spike_times = torch.randint(
                0, self.seq_length, (self.num_samples, self.n_input), generator=g)
            inputs = torch.zeros(self.num_samples, self.seq_length, self.n_input)
            sample_idx = torch.arange(self.num_samples).unsqueeze(1).expand(-1, self.n_input)
            neuron_idx = torch.arange(self.n_input).unsqueeze(0).expand(self.num_samples, -1)
            inputs[sample_idx, spike_times, neuron_idx] = self.input_gain
        elif task_type == "spatial":
            values = torch.rand(self.num_samples, self.n_input, generator=g) * self.input_gain
            inputs = values.unsqueeze(1).expand(-1, self.seq_length, -1).clone()
        else:
            raise ValueError(f"Unknown task_type: {task_type!r}. Use 'temporal' or 'spatial'.")
        self.inputs = inputs
        self.labels = torch.randint(0, self.n_classes, (self.num_samples,), generator=g).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
