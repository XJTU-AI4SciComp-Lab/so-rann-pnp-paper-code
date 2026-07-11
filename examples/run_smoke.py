"""Minimal runnable example for the SO-RaNN core utilities."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from RNN_ref import RNN_funs, err_calculate, sampling_points
from so_rann.reproducibility import set_seed


def main() -> None:
    """Run a tiny deterministic RaNN forward pass and error calculation."""
    set_seed(42)
    points = sampling_points.get_GL_points_2d(4)
    weights = sampling_points.get_weights_2d(4).reshape(-1, 1)

    network = RNN_funs.rnn_gauss_exact_1d(2, 8, 1)
    RNN_funs.weights_init_uniform(network, -1.0, 1.0)

    _, prediction, _, _ = network(points)
    reference = torch.zeros_like(prediction)
    error = err_calculate.err_L2_2d(prediction, reference, weights, -1, 1, -1, 1)

    print(f"points: {tuple(points.shape)}")
    print(f"relative smoke error: {error.item():.6e}")


if __name__ == "__main__":
    main()

