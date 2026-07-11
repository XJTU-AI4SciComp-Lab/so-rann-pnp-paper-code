from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.paths import legacy_output_path

# EX1（accuracy test）（SIAM-JSC-2021-ex2）


from RNN_ref import RNN_funs
from RNN_ref import sampling_points
import torch
import numpy as np
import argparse
import PNPNS_assemble_st_re_ex2
import copy
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
    parser.add_argument('-Nt', '--numbers_t', default=10, help='points of time', type=int)
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
    t2 = 1
    threshold = 1e-6
    D1 = 1
    D2 = 1
    z1 = 1
    z2 = -1
    eps = 1
    nv = 0.01

    # RaNN
    # c1m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    # c1m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
    c1m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c1m, -r_c1, r_c1)

    # RaNN
    # c2m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    # SO-RaNN
    # c2m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
    c2m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(c2m, -r_c2, r_c2)

    phim = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(phim, -r_phi, r_phi)

    um = RNN_funs.rnn_gauss_exact_2d_vec(3, m, 1).to(device)
    RNN_funs.weights_init_uniform_0(um, -r_u, r_u)
    RNN_funs.div_free_2d_gauss(um, m)

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

    scaler = PNPNS_assemble_st_re_ex2.pnpns_ex2_picard_st_decoupled_remass_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, tt, p1)

    um_0 = copy.deepcopy(um)
    um_3 = copy.deepcopy(um)

    c1m_0 = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c1m_0.pth'))
    c2m_0 = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c2m_0.pth'))
    um_0.load_state_dict(torch.load(legacy_output_path(f'PNPNS_code/model_ex2/um_weights_0.pth')))

    c1m_3 = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c1m_{Nt - 1}.pth'))
    c2m_3 = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c2m_{Nt - 1}.pth'))
    um_3.load_state_dict(torch.load(legacy_output_path(f'PNPNS_code/model_ex2/um_weights_{Nt - 1}.pth')))

    x1 = torch.linspace(a, b, 200)
    x2 = torch.linspace(c, d, 200)
    Y, X = torch.meshgrid(x2, x1) 
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    Z0 = torch.cat((Z, torch.zeros(40000, 1)), dim=1).to(device)
    Z1 = torch.cat((Z, 0.1 * torch.ones(40000, 1)), dim=1).to(device)
    Z2 = torch.cat((Z, 0.6 * torch.ones(40000, 1)), dim=1).to(device)
    Z3 = torch.cat((Z, 1 * torch.ones(40000, 1)), dim=1).to(device)

    pred1 = (scaler['scaler_c1'][0] * c1m_0(Z0)[1]).detach().cpu().numpy()
    pred5 = (scaler['scaler_c2'][0] * c2m_0(Z0)[1]).detach().cpu().numpy()
    pred9 = (um_0(Z0)[3]).detach().cpu().numpy()
    pred13 = (um_0(Z0)[4]).detach().cpu().numpy()

    pred4 = (scaler['scaler_c1'][-1] * c1m_3(Z3)[1]).detach().cpu().numpy()
    pred8 = (scaler['scaler_c2'][-1] * c2m_3(Z3)[1]).detach().cpu().numpy()
    pred12 = (um_3(Z3)[3]).detach().cpu().numpy()
    pred16 = (um_3(Z3)[4]).detach().cpu().numpy()

    t0 = t1
    for i in range(Nt):
        if t0 - 1e-3 < 0.1 <= t0 + s - 1e-3:
            c1m = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c1m_{i}.pth'))
            c2m = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c2m_{i}.pth'))
            um.load_state_dict(torch.load(legacy_output_path(f'PNPNS_code/model_ex2/um_weights_{i}.pth')))

            index1 = int((0.1 - t0) / s * Nre)
            pred2 = (scaler['scaler_c1'][i * Nre + index1] * c1m(Z1)[1]).detach().cpu().numpy()
            pred6 = (scaler['scaler_c2'][i * Nre + index1] * c2m(Z1)[1]).detach().cpu().numpy()
            pred10 = (um(Z1)[3]).detach().cpu().numpy()
            pred14 = (um(Z1)[4]).detach().cpu().numpy()

            print(i)
        if t0 - 1e-3 < 0.6 <= t0 + s - 1e-3:
            c1m = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c1m_{i}.pth'))
            c2m = torch.load(legacy_output_path(f'PNPNS_code/model_ex2/c2m_{i}.pth'))
            um.load_state_dict(torch.load(legacy_output_path(f'PNPNS_code/model_ex2/um_weights_{i}.pth')))

            index2 = int((0.6 - t0) / s * Nre)
            pred3 = (scaler['scaler_c1'][i * Nre + index2] * c1m(Z2)[1]).detach().cpu().numpy()
            pred7 = (scaler['scaler_c2'][i * Nre + index2] * c2m(Z2)[1]).detach().cpu().numpy()
            pred11 = (um(Z2)[3]).detach().cpu().numpy()
            pred15 = (um(Z2)[4]).detach().cpu().numpy()

            print(i)
        t0 = t0 + s

    plt.figure()
    ax1 = plt.subplot(4, 4, 1) 
    ax2 = plt.subplot(4, 4, 2)
    ax3 = plt.subplot(4, 4, 3)
    ax4 = plt.subplot(4, 4, 4)
    ax5 = plt.subplot(4, 4, 5)
    ax6 = plt.subplot(4, 4, 6)
    ax7 = plt.subplot(4, 4, 7)
    ax8 = plt.subplot(4, 4, 8)
    ax9 = plt.subplot(4, 4, 9)
    ax10 = plt.subplot(4, 4, 10)
    ax11 = plt.subplot(4, 4, 11)
    ax12 = plt.subplot(4, 4, 12)
    ax13 = plt.subplot(4, 4, 13)
    ax14 = plt.subplot(4, 4, 14)
    ax15 = plt.subplot(4, 4, 15)
    ax16 = plt.subplot(4, 4, 16)

    plt.tight_layout() 
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15) 

    plt.sca(ax1)
    pred1 = pred1.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred1, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax1.set_title('Exact c1', fontsize=10)

    plt.sca(ax2)
    pred2 = pred2.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred2, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax2.set_title('Numerical c1', fontsize=10)


    plt.sca(ax3)
    pred3 = pred3.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred3, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax3.set_title('abs_error c1', fontsize=10)

    plt.sca(ax4)
    pred4 = pred4.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred4, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax4.set_title('Exact c2', fontsize=10)

    plt.sca(ax5)
    pred5 = pred5.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred5, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax5.set_title('Numerical c2', fontsize=10)

    plt.sca(ax6)
    pred6 = pred6.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred6, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax6.set_title('abs_error c2', fontsize=10)

    plt.sca(ax7)
    pred7 = pred7.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred7, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax7.set_title('Exact u1', fontsize=10)

    plt.sca(ax8)
    pred8 = pred8.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred8, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=0, vmax=2.5) 
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax8.set_title('Numerical u1', fontsize=10)

    plt.sca(ax9)
    pred9 = pred9.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred9, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax9)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax9.set_title('abs_error u1', fontsize=10)

    plt.sca(ax10)
    pred10 = pred10.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred10, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax10)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax10.set_title('Exact u2', fontsize=10)

    plt.sca(ax11)
    pred11 = pred11.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred11, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax11)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax11.set_title('Numerical u2', fontsize=10)

    plt.sca(ax12)
    pred12 = pred12.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred12, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax12)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax12.set_title('abs_error u2', fontsize=10)

    plt.sca(ax13)
    pred13 = pred13.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred13, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax13)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax9.set_title('abs_error u1', fontsize=10)

    plt.sca(ax14)
    pred14 = pred14.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred14, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax14)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax10.set_title('Exact u2', fontsize=10)

    plt.sca(ax15)
    pred15 = pred15.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred15, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax15)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax11.set_title('Numerical u2', fontsize=10)

    plt.sca(ax16)
    pred16 = pred16.reshape(200, 200)
    cmap = matplotlib.cm.jet
    cmap.set_bad(color='w')
    h = plt.imshow(pred16, interpolation='nearest', cmap=cmap,
                   extent=[a, b, c, d],
                   origin='lower', aspect=1, vmin=-4, vmax=4) 
    divider = make_axes_locatable(ax16)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    # ax12.set_title('abs_error u2', fontsize=10)

    plt.rcParams['font.sans-serif'] = ['Times New Roman'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.savefig(legacy_output_path('PNPNS_code/PNPNS_results_ex2/PNPNS-st-re-ex2-p.eps'))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = get_args() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    main(r_c1=args.range1,r_c2=args.range2,r_phi=args.range3,r_u=args.range4,r_p=args.range5, n=args.numbers1, Nt=args.numbers_t, Nre=args.numbers_re, n_int=args.numbers2, m=args.neurons, tt=args.times, p1=args.penalty1, device=device)
