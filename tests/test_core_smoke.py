from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from RNN_ref import RNN_funs, err_calculate, sampling_points
from so_rann.paths import legacy_output_path
from so_rann.reproducibility import set_seed


def test_gauss_legendre_shapes_and_weights():
    points = sampling_points.get_GL_points_2d(3)
    weights = sampling_points.get_weights_2d(3)

    assert points.shape == (9, 2)
    assert weights.shape == (9,)
    assert torch.isclose(weights.sum(), torch.tensor(4.0))


def test_small_rann_forward_is_deterministic():
    set_seed(123)
    model = RNN_funs.rnn_gauss_exact_1d(2, 5, 1)
    RNN_funs.weights_init_uniform(model, -0.5, 0.5)
    x = torch.linspace(-1.0, 1.0, 10).reshape(5, 2)

    hidden, output, dx, dx2 = model(x)

    assert hidden.shape == (5, 5)
    assert output.shape == (5, 1)
    assert dx.shape == (5, 5)
    assert dx2.shape == (5, 5)


def test_quadrature_error_and_legacy_paths():
    points = sampling_points.get_GL_points_2d(2)
    weights = sampling_points.get_weights_2d(2).reshape(-1, 1)
    values = points[:, 0:1] * 0.0

    error = err_calculate.err_L2_2d(values, values, weights, -1, 1, -1, 1)
    path = legacy_output_path("PNP_code/model_ex3/c1m_0.pth")

    assert torch.isclose(error, torch.tensor(0.0))
    assert path.parts[-4:] == ("checkpoints", "PNP_code", "model_ex3", "c1m_0.pth")


if __name__ == "__main__":
    test_gauss_legendre_shapes_and_weights()
    test_small_rann_forward_is_deterministic()
    test_quadrature_error_and_legacy_paths()
    print("core smoke tests passed")
