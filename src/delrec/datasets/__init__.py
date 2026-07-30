"""Dataset loaders. :func:`delrec.datasets.load_dataset` dispatches on
``config.dataset`` and returns the (train, valid, test) dataloaders."""

from .loaders import load_dataset

__all__ = ["load_dataset"]
