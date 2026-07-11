from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# EX2（JCP-2019 ex.2）（benchmark test）


from RNN_ref import RNN_funs
from RNN_ref import sampling_points
import torch
import numpy as np
import argparse
import PNP_assemble_st_re_ex2
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(42)
np.random.seed(42)


def get_args(): 
    parser = argparse.ArgumentParser(description='Train the pde net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r_c1', '--range1', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r_c2', '--range2', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r_phi', '--range3', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-n', '--numbers1', default=20, help='points on unit edge', type=int)
    parser.add_argument('-Nt', '--numbers_t', default=1, help='points of time', type=int)
    parser.add_argument('-Nre', '--numbers_re', default=51, help='points of time', type=int)
    parser.add_argument('-n_int', '--numbers2', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=400, help='number of hidden layer nodes', type=int)
    parser.add_argument('-tt', '--times', default=30, help='times of Picard iteration', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    return parser.parse_args()


def main(r_c1, r_c2, r_phi, n, Nt, Nre, n_int, m, tt, p1, device): 
    a = 0
    b = 1
    c = 0
    d = 1
    t1 = 0
    t2 = 1
    threshold = 1e-6
    D1 = 1
    D2 = 1
    z1 = 1
    z2 = -1
    eps = 1

    # RaNN
    # c1m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    c1m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c1m, -r_c1, r_c1)

    # RaNN
    # c2m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    c2m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c2m, -r_c2, r_c2)
    # phim = RNN_funs.rnn_sin_exact_1d(3, m, 1).to(device)

    phim = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(phim, -r_phi, r_phi)

    xr2 = sampling_points.get_GL_points_2d(n_int)
    wr2 = sampling_points.get_weights_2d(n_int)
    wr2 = wr2.reshape([n_int ** 2, 1])
    wr2 = wr2.to(device)
    xr2 = xr2.to(device)
    x2 = torch.zeros_like(xr2)
    x2[:, 0:1] = (b - a) / 2 * xr2[:, 0:1] + (a + b) / 2
    x2[:, 1:2] = (d - c) / 2 * xr2[:, 1:2] + (c + d) / 2

    s = (t2 - t1) / Nt
    # s = 1

    # SO-RaNN
    PNP_assemble_st_re_ex2.pnp_ex2_picard_st_remass_G_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, tt, p1)
    # RaNN
    # PNP_assemble_st_re_ex2.pnp_ex2_picard_st(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, tt, p1)


if __name__ == '__main__':
    args = get_args() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    main(r_c1=args.range1, r_c2=args.range2, r_phi=args.range3, n=args.numbers1, Nt=args.numbers_t, Nre=args.numbers_re,
         n_int=args.numbers2, m=args.neurons, tt=args.times, p1=args.penalty1, device=device)
