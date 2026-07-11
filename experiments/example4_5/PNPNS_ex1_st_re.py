from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# PNPNS
# EX5（accuracy test）（SIAM-JSC-2021-ex2）


from RNN_ref import RNN_funs
from RNN_ref import sampling_points
import torch
import numpy as np
import argparse
import PNPNS_assemble_st_re_ex1
import ex1_real
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(42)
np.random.seed(42)

def get_args(): 
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r_c1', '--range1', default=4, help='Range of initial weights', type=float)
    parser.add_argument('-r_c2', '--range2', default=4, help='Range of initial weights', type=float)
    parser.add_argument('-r_phi', '--range3', default=4, help='Range of initial weights', type=float)
    parser.add_argument('-r_u', '--range4', default=4, help='Range of initial weights', type=float)
    parser.add_argument('-r_p', '--range5', default=4, help='Range of initial weights', type=float)
    parser.add_argument('-n', '--numbers1', default=20, help='points on unit edge', type=int)
    parser.add_argument('-Nt', '--numbers_t', default=1, help='points of time', type=int)
    parser.add_argument('-Nre', '--numbers_re', default=51, help='points of time', type=int)
    parser.add_argument('-n_int', '--numbers2', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=400, help='number of hidden layer nodes', type=int)
    parser.add_argument('-tt', '--times', default=20, help='times of Picard iteration', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    return parser.parse_args()

def main(r_c1,r_c2,r_phi,r_u,r_p,n,Nt,Nre,n_int,m,tt,p1,device):
    a = -1
    b = 1
    c = -1
    d = 1
    t1 = 0
    t2 = 0.1
    threshold = 1e-6
    D1 = 1
    D2 = 1
    z1 = 1
    z2 = -1
    eps = 1
    nv = 0.001

    # c1
    # RaNN
    # c1m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    # c1m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
    c1m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c1m, -r_c1, r_c1)

    # c2
    # RaNN
    # c2m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    # c2m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
    c2m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c2m, -r_c2, r_c2)

    # phi
    phim = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(phim, -r_phi, r_phi)

    # v
    um = RNN_funs.rnn_gauss_exact_2d_vec(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(um, -r_u, r_u)
    RNN_funs.div_free_2d_gauss(um, m)

    # p
    pm = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(pm, -r_p, r_p)

    xr2 = sampling_points.get_GL_points_2d(n_int)
    wr2 = sampling_points.get_weights_2d(n_int)
    wr2 = wr2.reshape([n_int ** 2, 1])
    wr2 = wr2.to(device)
    xr2 = xr2.to(device)
    x2 = torch.zeros_like(xr2)
    x2[:, 0:1] = (b - a) / 2 * xr2[:, 0:1] + (a + b) / 2
    x2[:, 1:2] = (d - c) / 2 * xr2[:, 1:2] + (c + d) / 2

    s = (t2 - t1)/Nt
    # s = 1

    # SO-RaNN
    PNPNS_assemble_st_re_ex1.pnpns_ex1_picard_st_decoupled_remass(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, tt, p1)
    # RaNN
    # PNPNS_assemble_st_re_ex1.pnpns_ex1_picard_st_decoupled(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, tt, p1)

if __name__ == '__main__':
    args = get_args() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    main(r_c1=args.range1,r_c2=args.range2,r_phi=args.range3,r_u=args.range4,r_p=args.range5, n=args.numbers1, Nt=args.numbers_t, Nre=args.numbers_re, n_int=args.numbers2, m=args.neurons, tt=args.times, p1=args.penalty1, device=device)
