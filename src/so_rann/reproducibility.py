"""Reproducibility helpers shared by examples and tests."""

import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
