from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.paths import legacy_output_path

import ex4_real
import numpy as np
from RNN_ref import err_calculate
from RNN_ref import RNN_funs
from RNN_ref import plot_results
from RNN_ref import fd
import torch
from RNN_ref import sampling_points
import scipy
import time

def PNP_plot_st_re_G_benchmark(xt_re, results, k):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })

    fig, axes = plt.subplots(1, 5, figsize=(10, 2), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5 = axes.flat

    ax1.plot(xt_re, results['c1_min'])
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('minimum')
    ax1.set_title(r'$c_1$ min')
    ax1.grid(True, alpha=0.3)

    ax2.plot(xt_re, results['c2_min'])
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('minimum')
    ax2.set_title(r'$c_2$ min')
    ax2.grid(True, alpha=0.3)

    ax3.plot(xt_re, results['c3_min'])
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('minimum')
    ax3.set_title(r'$c_3$ min')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['m1'], label=r'$m_1$', color='green', linestyle='-')
    ax4.plot(xt_re, results['m2'], label=r'$m_2$', color='red', linestyle='--')
    ax4.plot(xt_re, results['m3'], label=r'$m_3$', color='blue', linestyle=':')
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('mass')
    ax4.set_title(r'$m_1$, $m_2$ and $m_3$')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax5.plot(xt_re, results['e'])
    ax5.set_xlabel('$t$')
    ax5.set_ylabel('$E$')
    ax5.set_title('energy')
    ax5.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex4_paper/PNP-st-re-G-ex4_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_G_all_benchmark(xt_re, results):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)

    ax1, ax2, ax3 = axes[0, :] 
    ax4, ax5 = axes[1, 0], axes[1, 1] 
    axes[1, 2].set_visible(False)

    ax1.plot(xt_re, results['c1_min_all'])
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('minimum')
    ax1.set_title(r'$c_1$ min')
    ax1.grid(True, alpha=0.3)

    ax2.plot(xt_re, results['c2_min_all'])
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('minimum')
    ax2.set_title(r'$c_2$ min')
    ax2.grid(True, alpha=0.3)

    ax3.plot(xt_re, results['c3_min_all'])
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('minimum')
    ax3.set_title(r'$c_3$ min')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['m1_all'], label=r'$m_1$', color='green', linestyle='-')
    ax4.plot(xt_re, results['m2_all'], label=r'$m_2$', color='red', linestyle='--')
    ax4.plot(xt_re, results['m3_all'], label=r'$m_3$', color='blue', linestyle=':')
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('mass')
    ax4.set_title(r'$m_1$, $m_2$ and $m_3$')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax5.plot(xt_re, results['e_all'])
    ax5.set_xlabel('$t$')
    ax5.set_ylabel('$E$')
    ax5.set_title('energy')
    ax5.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex4_paper/PNP-st-re-G-all-ex4.eps'),
                bbox_inches='tight')
    plt.close(fig)

def pnp_ex4_picard_st_remass_G_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, c3m, phim, a, b, c, d, t1, D1, D2, D3, z1,
                               z2, z3, eps, threshold, ite, p1):
    device = torch.device('cpu')
    total_time1 = 0 
    total_time2 = 0 
    total_time3 = 0 
    c0 = 1

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'm3': np.zeros(Nre),
        'e_bar': np.zeros(Nre), 'e': np.zeros(Nre),
        'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre), 'c3_min': np.zeros(Nre)
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt), 'm3_all': np.zeros(Nre * Nt),
        'e_bar_all': np.zeros(Nre * Nt), 'e_all': np.zeros(Nre * Nt),
        'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt), 'c3_min_all': np.zeros(Nre * Nt)
    }

    start1 = time.perf_counter()

    xr = sampling_points.get_GL_points_3d(n_int)
    wr = sampling_points.get_weights_3d(n_int).reshape([n_int ** 3, 1])
    x = torch.zeros_like(xr)
    x[:, 0:1] = (b - a) / 2 * xr[:, 0:1] + (a + b) / 2
    x[:, 1:2] = (d - c) / 2 * xr[:, 1:2] + (c + d) / 2

    x2_0 = torch.cat((x2, torch.zeros(x2.shape[0], 1)), dim=1)
    with torch.no_grad():
        mass1 = err_calculate.int_2d(ex4_real.c1(x2_0), wr2, a, b, c, d)
        mass2 = err_calculate.int_2d(ex4_real.c2(x2_0), wr2, a, b, c, d)
        mass3 = err_calculate.int_2d(ex4_real.c3(x2_0), wr2, a, b, c, d)

    t2 = t1 + s
    RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(c3m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)

    N1 = int((d - c) * (b - a) * s * n ** 3)
    N2x = int((b - a) * s * n ** 2)
    N2y = int((d - c) * s * n ** 2)
    N3 = int((d - c) * (b - a) * n ** 2)
    xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
    xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
    xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
    xt_re = torch.linspace(t1, t2, Nre)
    x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

    nor = torch.zeros(2 * (N2x + N2y), 2)
    nor[0:N2x, 1:2] = -1
    nor[N2x:2 * N2x, 1:2] = 1
    nor[2 * N2x:2 * N2x + N2y, 0:1] = -1
    nor[2 * N2x + N2y:2 * (N2x + N2y), 0:1] = 1

    n1, n2 = nor[:, 0:1], nor[:, 1:2]

    with torch.no_grad():
        fa1 = np.zeros((N1, 1))
        fa2 = np.zeros((N1, 1))
        fa3 = np.zeros((N1, 1))
        fb1 = np.zeros((2 * (N2x + N2y), 1))
        fb2 = np.zeros((2 * (N2x + N2y), 1))
        fb3 = np.zeros((2 * (N2x + N2y), 1))
        fb4 = np.zeros((2 * (N2x + N2y), 1))
        fi1 = (ex4_real.c1(xi)).detach().cpu().numpy()
        fi2 = (ex4_real.c2(xi)).detach().cpu().numpy()
        fi3 = (ex4_real.c3(xi)).detach().cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_phi = phim(x2_t)[0]
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()

    c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(
        c1m, m, xy_inter)
    c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(
        c2m, m, xy_inter)
    c3m_in, c3md1_in, c3md2_in, c3mdx_in, c3mdy_in, c3mdt_in, c3mdxx_in, c3mdyy_in = RNN_funs.compute_nn_derivatives_1d(
        c3m, m, xy_inter)
    phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(
        phim, m, xy_inter)

    c1md1_b, c2md1_b, c3md1_b, phimd1_b = c1m(xb)[2], c2m(xb)[2], c3m(xb)[2], phim(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    c3mdx_b = c3m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c3md1_b
    c3mdy_b = c3m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c3md1_b

    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    c1m_i, c2m_i, c3m_i = c1m(xi)[0], c2m(xi)[0], c3m(xi)[0]

    end1 = time.perf_counter()
    total_time1 += (end1 - start1)

    diffc1, diffc2, diffc3, diffphi = 1, 1, 1, 1
    diff_list = {'c1': [], 'c2': [], 'c3': [], 'phi': []}
    ttt = 0

    while (diffc1 > threshold or diffc2 > threshold or diffc3 > threshold or diffphi > threshold) and ttt < ite:
        start2 = time.perf_counter()
        c1_before, c2_before, c3_before, phi_before = c1m(x)[1], c2m(x)[1], c3m(x)[1], phim(x)[1]

        # NP
        phimdx_in_1 = phimdx_in @ phim.predict.weight.data.T
        phimdy_in_1 = phimdy_in @ phim.predict.weight.data.T
        phimdxx_in_1 = phimdxx_in @ phim.predict.weight.data.T
        phimdyy_in_1 = phimdyy_in @ phim.predict.weight.data.T

        A1 = (c1mdt_in - D1 * (c1mdxx_in + c1mdyy_in) - D1 * z1 * (
                    c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1 + c1m_in * (
                        phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
        b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt_in - D2 * (c2mdxx_in + c2mdyy_in) - D2 * z2 * (
                    c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1 + c2m_in * (
                        phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
        b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        A3 = (c3mdt_in - D2 * (c3mdxx_in + c3mdyy_in) - D2 * z2 * (
                    c3mdx_in * phimdx_in_1 + c3mdy_in * phimdy_in_1 + c3m_in * (
                        phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B3 = (c3mdx_b * n1 + c3mdy_b * n2).detach().cpu().numpy()
        I3 = (c3m_i).detach().cpu().numpy()

        A_NP3 = np.vstack((A3, p1 * B3, p1 * I3)) 
        b_NP3 = np.vstack((fa3, p1 * fb3, p1 * fi3))
        wu_NP3 = np.linalg.lstsq(A_NP3, b_NP3)[0] 
        c3m.predict.weight.data = torch.from_numpy(wu_NP3[0:m]).T.to(device)

        # P
        c1m_in_1 = c1m(xy_inter)[1]
        c2m_in_1 = c2m(xy_inter)[1]
        c3m_in_1 = c3m(xy_inter)[1]

        A4 = -eps ** 2 * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
        fa4 = (z1 * c1m_in_1 + z2 * c2m_in_1 + z3 * c3m_in_1).detach().cpu().numpy()
        B4 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

        A_P = np.vstack((A4, p1 * B4, E1)) 
        b_P = np.vstack((fa4, p1 * fb4, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        c1_after, c2_after, c3_after, phi_after = c1m(x)[1], c2m(x)[1], c3m(x)[1], phim(x)[1]
        diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
        diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
        diffc3 = err_calculate.err_L2_3d(c3_after, c3_before, wr, a, b, c, d, t1, t2)
        diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
        diff_list['c1'].append(float(diffc1.item()))
        diff_list['c2'].append(float(diffc2.item()))
        diff_list['c3'].append(float(diffc3.item()))
        diff_list['phi'].append(float(diffphi.item()))

        ttt = ttt + 1

        end2 = time.perf_counter()
        total_time2 += (end2 - start2)

    print(ttt)

    start3 = time.perf_counter()

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)
    mass3_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)
            mass3_re[i] = err_calculate.int_2d(c3m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re
    s_c3 = mass3 / mass3_re

    s_phi = torch.ones(Nre) 
    phid1_0 = phim(x2_0)[2]
    phidx_0 = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
    phidy_0 = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
    r_0 = err_calculate.int_2d(ex4_real.c1(x2_0) * torch.log(ex4_real.c1(x2_0)) + ex4_real.c2(x2_0) * torch.log(
        ex4_real.c2(x2_0)) + ex4_real.c3(x2_0) * torch.log(ex4_real.c3(x2_0)) + eps ** 2 / 2 * (
                                           phidx_0 ** 2 + phidy_0 ** 2), wr2, a, b, c, d) + c0
    results['e_bar'][0] = r_0
    ss = s / (Nre - 1)
    for i in range(Nre - 1):
        x2_re = torch.cat((x2, xt_re[i + 1] * torch.ones(x2.shape[0], 1)), dim=1)
        c1m_re = s_c1[i + 1] * c1m(x2_re)[1]
        c2m_re = s_c2[i + 1] * c2m(x2_re)[1]
        c3m_re = s_c3[i + 1] * c3m(x2_re)[1]
        c1md1_re = c1m(x2_re)[2]
        c2md1_re = c2m(x2_re)[2]
        c3md1_re = c3m(x2_re)[2]
        phid1_re = phim(x2_re)[2]
        c1dx_re = s_c1[i + 1] * (
                    (c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_re) @ c1m.predict.weight.data.T)
        c1dy_re = s_c1[i + 1] * (
                    (c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re) @ c1m.predict.weight.data.T)
        c2dx_re = s_c2[i + 1] * (
                    (c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_re) @ c2m.predict.weight.data.T)
        c2dy_re = s_c2[i + 1] * (
                    (c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re) @ c2m.predict.weight.data.T)
        c3dx_re = s_c3[i + 1] * (
                    (c3m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c3md1_re) @ c3m.predict.weight.data.T)
        c3dy_re = s_c3[i + 1] * (
                    (c3m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c3md1_re) @ c3m.predict.weight.data.T)
        phidx_re = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_re) @ phim.predict.weight.data.T
        phidy_re = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_re) @ phim.predict.weight.data.T

        E_n1 = err_calculate.int_2d(
            c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + c3m_re * torch.log(c3m_re) + eps ** 2 / 2 * (
                        phidx_re ** 2 + phidy_re ** 2), wr2, a, b, c, d) + c0
        Edt = err_calculate.int_2d(D1 * c1m_re * (
                    (c1dx_re / c1m_re + z1 * phidx_re) ** 2 + (c1dy_re / c1m_re + z1 * phidy_re) ** 2) + D2 * c2m_re * (
                                               (c2dx_re / c2m_re + z2 * phidx_re) ** 2 + (
                                                   c2dy_re / c2m_re + z2 * phidy_re) ** 2) + D3 * c3m_re * (
                                               (c3dx_re / c3m_re + z3 * phidx_re) ** 2 + (
                                                   c3dy_re / c3m_re + z3 * phidy_re) ** 2), wr2, a, b, c, d)
        results['e_bar'][i + 1] = results['e_bar'][i] / (1 + ss * Edt / E_n1)
        s_phi[i + 1] = results['e_bar'][i + 1] / E_n1

    xt_re_np = xt_re.squeeze().numpy()
    s_phi_np = s_phi.detach().numpy()
    s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
    s_phi_in_0 = torch.from_numpy(s_phi_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)


    # NP
    phimdx_in_1 = s_phi_in_0 * phimdx_in @ phim.predict.weight.data.T
    phimdy_in_1 = s_phi_in_0 * phimdy_in @ phim.predict.weight.data.T
    phimdxx_in_1 = s_phi_in_0 * phimdxx_in @ phim.predict.weight.data.T
    phimdyy_in_1 = s_phi_in_0 * phimdyy_in @ phim.predict.weight.data.T

    A1 = (c1mdt_in - D1 * (c1mdxx_in + c1mdyy_in) - D1 * z1 * (
                c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1 + c1m_in * (
                    phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
    B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
    I1 = (c1m_i).detach().cpu().numpy()

    A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
    b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
    wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
    c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

    A2 = (c2mdt_in - D2 * (c2mdxx_in + c2mdyy_in) - D2 * z2 * (
                c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1 + c2m_in * (
                    phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
    B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
    I2 = (c2m_i).detach().cpu().numpy()

    A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
    b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
    wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
    c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

    A3 = (c3mdt_in - D2 * (c3mdxx_in + c3mdyy_in) - D2 * z2 * (
                c3mdx_in * phimdx_in_1 + c3mdy_in * phimdy_in_1 + c3m_in * (
                    phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
    B3 = (c3mdx_b * n1 + c3mdy_b * n2).detach().cpu().numpy()
    I3 = (c3m_i).detach().cpu().numpy()

    A_NP3 = np.vstack((A3, p1 * B3, p1 * I3)) 
    b_NP3 = np.vstack((fa3, p1 * fb3, p1 * fi3))
    wu_NP3 = np.linalg.lstsq(A_NP3, b_NP3)[0] 
    c3m.predict.weight.data = torch.from_numpy(wu_NP3[0:m]).T.to(device)

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)
    mass3_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)
            mass3_re[i] = err_calculate.int_2d(c3m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re
    s_c3 = mass3 / mass3_re

    xt_re_np = xt_re.squeeze().numpy()
    s_c1_np = s_c1.squeeze().detach().numpy()
    s_c2_np = s_c2.squeeze().detach().numpy()
    s_c3_np = s_c3.squeeze().detach().numpy()

    s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
    s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
    s_c3_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c3_np, kind='linear', fill_value='extrapolate')
    s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    s_c3_in_0 = torch.from_numpy(s_c3_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # P
    c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
    c2m_in_1 = s_c1_in_0 * c2m(xy_inter)[1]
    c3m_in_1 = s_c1_in_0 * c3m(xy_inter)[1]

    A4 = -eps ** 2 * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
    fa4 = (z1 * c1m_in_1 + z2 * c2m_in_1 + z3 * c3m_in_1).detach().cpu().numpy()
    B4 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

    A_P = np.vstack((A4, p1 * B4, E1)) 
    b_P = np.vstack((fa4, p1 * fb4, fe1))
    wu_P = np.linalg.lstsq(A_P, b_P)[0] 
    phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

    end3 = time.perf_counter()
    total_time3 += (end3 - start3)

    for i in range(Nre):
        x2_plot = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        c1m_plot = s_c1[i] * c1m(x2_plot)[1]
        c2m_plot = s_c2[i] * c2m(x2_plot)[1]
        c3m_plot = s_c3[i] * c3m(x2_plot)[1]
        results['c1_min'][i] = torch.min(c1m_plot)
        results['c2_min'][i] = torch.min(c2m_plot)
        results['c3_min'][i] = torch.min(c3m_plot)
        results['m1'][i] = err_calculate.int_2d(s_c1[i] * c1m(x2_plot)[1], wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(s_c2[i] * c2m(x2_plot)[1], wr2, a, b, c, d)
        results['m3'][i] = err_calculate.int_2d(s_c3[i] * c3m(x2_plot)[1], wr2, a, b, c, d)
        phid1_plot = phim(x2_plot)[2]
        phidx_plot = ((phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_plot) @ phim.predict.weight.data.T)
        phidy_plot = ((phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_plot) @ phim.predict.weight.data.T)
        energy_terms = c1m_plot * torch.log(c1m_plot) + c2m_plot * torch.log(c2m_plot) + c3m_plot * torch.log(
            c3m_plot) + eps ** 2 / 2 * (phidx_plot ** 2 + phidy_plot ** 2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['e_bar_all'][0:Nre] = results['e_bar']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['m3_all'][0:Nre] = results['m3']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['c3_min_all'][0:Nre] = results['c3_min']

    PNP_plot_st_re_G_benchmark(xt_re, results, 0)

    t1 = t2

    for k in range(Nt - 1):
        start1 = time.perf_counter()

        t2 = t1 + s
        x2_0 = torch.cat((x2, t1 * torch.ones(x2.shape[0], 1)), dim=1)

        results = {
            'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'm3': np.zeros(Nre),
            'e_bar': np.zeros(Nre), 'e': np.zeros(Nre),
            'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre), 'c3_min': np.zeros(Nre)
        }

        c1m_in_n = s_c1[-1] * c1m(x2_0)[1]
        c2m_in_n = s_c2[-1] * c2m(x2_0)[1]
        c3m_in_n = s_c3[-1] * c3m(x2_0)[1]
        phidx_int_n = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        phidy_int_n = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        r_n = err_calculate.int_2d(
            c1m_in_n * torch.log(c1m_in_n) + c2m_in_n * torch.log(c2m_in_n) + c3m_in_n * torch.log(
                c3m_in_n) + eps ** 2 / 2 * (phidx_int_n ** 2 + phidy_int_n ** 2), wr2, a, b, c, d) + c0
        results['e_bar'][0] = r_n

        start = time.perf_counter()

        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        with torch.no_grad():
            fa1 = np.zeros((N1, 1))
            fa2 = np.zeros((N1, 1))
            fa3 = np.zeros((N1, 1))
            fb1 = np.zeros((2 * (N2x + N2y), 1))
            fb2 = np.zeros((2 * (N2x + N2y), 1))
            fb3 = np.zeros((2 * (N2x + N2y), 1))
            fb4 = np.zeros((2 * (N2x + N2y), 1))
            fi1 = (s_c1[-1] * c1m(xi)[1]).detach().cpu().numpy()
            fi2 = (s_c2[-1] * c2m(xi)[1]).detach().cpu().numpy()
            fi3 = (s_c3[-1] * c3m(xi)[1]).detach().cpu().numpy()

        RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c3m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_phi = phim(x2_t)[0]
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()

        c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(
            c1m, m, xy_inter)
        c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(
            c2m, m, xy_inter)
        c3m_in, c3md1_in, c3md2_in, c3mdx_in, c3mdy_in, c3mdt_in, c3mdxx_in, c3mdyy_in = RNN_funs.compute_nn_derivatives_1d(
            c3m, m, xy_inter)
        phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(
            phim, m, xy_inter)

        c1md1_b, c2md1_b, c3md1_b, phimd1_b = c1m(xb)[2], c2m(xb)[2], c3m(xb)[2], phim(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        c3mdx_b = c3m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c3md1_b
        c3mdy_b = c3m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c3md1_b

        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        c1m_i, c2m_i, c3m_i = c1m(xi)[0], c2m(xi)[0], c3m(xi)[0]

        end1 = time.perf_counter()
        total_time1 += (end1 - start1)

        start2 = time.perf_counter()

        diffc1, diffc2, diffc3, diffphi = 1, 1, 1, 1
        diff_list = {'c1': [], 'c2': [], 'c3': [], 'phi': []}
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffc3 > threshold or diffphi > threshold) and ttt < ite:
            c1_before = c1m(x)[1]
            c2_before = c2m(x)[1]
            c3_before = c3m(x)[1]
            phi_before = phim(x)[1]

            # NP
            phimdx_in_1 = phimdx_in @ phim.predict.weight.data.T
            phimdy_in_1 = phimdy_in @ phim.predict.weight.data.T
            phimdxx_in_1 = phimdxx_in @ phim.predict.weight.data.T
            phimdyy_in_1 = phimdyy_in @ phim.predict.weight.data.T

            A1 = (c1mdt_in - D1 * (c1mdxx_in + c1mdyy_in) - D1 * z1 * (
                        c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1 + c1m_in * (
                            phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
            B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
            b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt_in - D2 * (c2mdxx_in + c2mdyy_in) - D2 * z2 * (
                        c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1 + c2m_in * (
                            phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
            B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
            b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            A3 = (c3mdt_in - D2 * (c3mdxx_in + c3mdyy_in) - D2 * z2 * (
                        c3mdx_in * phimdx_in_1 + c3mdy_in * phimdy_in_1 + c3m_in * (
                            phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
            B3 = (c3mdx_b * n1 + c3mdy_b * n2).detach().cpu().numpy()
            I3 = (c3m_i).detach().cpu().numpy()

            A_NP3 = np.vstack((A3, p1 * B3, p1 * I3)) 
            b_NP3 = np.vstack((fa3, p1 * fb3, p1 * fi3))
            wu_NP3 = np.linalg.lstsq(A_NP3, b_NP3)[0] 
            c3m.predict.weight.data = torch.from_numpy(wu_NP3[0:m]).T.to(device)

            # P
            c1m_1_in = c1m(xy_inter)[1]
            c2m_1_in = c2m(xy_inter)[1]
            c3m_1_in = c3m(xy_inter)[1]

            A4 = -eps ** 2 * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
            fa4 = (z1 * c1m_1_in + z2 * c2m_1_in + z3 * c3m_1_in).detach().cpu().numpy()
            B4 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

            A_P = np.vstack((A4, p1 * B4, E1)) 
            b_P = np.vstack((fa4, p1 * fb4, fe1))
            wu_P = np.linalg.lstsq(A_P, b_P)[0] 
            phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

            c1_after, c2_after, c3_after, phi_after = c1m(x)[1], c2m(x)[1], c3m(x)[1], phim(x)[1]
            diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
            diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
            diffc3 = err_calculate.err_L2_3d(c3_after, c3_before, wr, a, b, c, d, t1, t2)
            diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
            diff_list['c1'].append(float(diffc1.item()))
            diff_list['c2'].append(float(diffc2.item()))
            diff_list['c3'].append(float(diffc3.item()))
            diff_list['phi'].append(float(diffphi.item()))

            ttt = ttt + 1

            end2 = time.perf_counter()
            total_time2 += (end2 - start2)

        print(ttt)

        start3 = time.perf_counter()

        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)
        mass3_re = torch.zeros(Nre)
        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)
                mass3_re[i] = err_calculate.int_2d(c3m(x2_re)[1], wr2, a, b, c, d)
        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re
        s_c3 = mass3 / mass3_re

        s_phi = torch.ones(Nre) 
        ss = s / (Nre - 1)
        for i in range(Nre - 1):
            x2_re = torch.cat((x2, xt_re[i + 1] * torch.ones(x2.shape[0], 1)), dim=1)
            c1m_re = s_c1[i + 1] * c1m(x2_re)[1]
            c2m_re = s_c2[i + 1] * c2m(x2_re)[1]
            c3m_re = s_c3[i + 1] * c3m(x2_re)[1]
            c1md1_re = c1m(x2_re)[2]
            c2md1_re = c2m(x2_re)[2]
            c3md1_re = c3m(x2_re)[2]
            phid1_re = phim(x2_re)[2]
            c1dx_re = s_c1[i + 1] * (
                        (c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_re) @ c1m.predict.weight.data.T)
            c1dy_re = s_c1[i + 1] * (
                        (c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re) @ c1m.predict.weight.data.T)
            c2dx_re = s_c2[i + 1] * (
                        (c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_re) @ c2m.predict.weight.data.T)
            c2dy_re = s_c2[i + 1] * (
                        (c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re) @ c2m.predict.weight.data.T)
            c3dx_re = s_c3[i + 1] * (
                        (c3m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c3md1_re) @ c3m.predict.weight.data.T)
            c3dy_re = s_c3[i + 1] * (
                        (c3m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c3md1_re) @ c3m.predict.weight.data.T)
            phidx_re = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_re) @ phim.predict.weight.data.T
            phidy_re = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_re) @ phim.predict.weight.data.T

            E_n1 = err_calculate.int_2d(
                c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + c3m_re * torch.log(c3m_re) + eps ** 2 / 2 * (
                            phidx_re ** 2 + phidy_re ** 2), wr2, a, b, c, d) + c0
            Edt = err_calculate.int_2d(D1 * c1m_re * ((c1dx_re / c1m_re + z1 * phidx_re) ** 2 + (
                        c1dy_re / c1m_re + z1 * phidy_re) ** 2) + D2 * c2m_re * (
                                                   (c2dx_re / c2m_re + z2 * phidx_re) ** 2 + (
                                                       c2dy_re / c2m_re + z2 * phidy_re) ** 2) + D3 * c3m_re * (
                                                   (c3dx_re / c3m_re + z3 * phidx_re) ** 2 + (
                                                       c3dy_re / c3m_re + z3 * phidy_re) ** 2), wr2, a, b, c, d)
            results['e_bar'][i + 1] = results['e_bar'][i] / (1 + ss * Edt / E_n1)
            s_phi[i + 1] = results['e_bar'][i + 1] / E_n1

        # NP
        phimdx_in_1 = s_phi_in_0 * phimdx_in @ phim.predict.weight.data.T
        phimdy_in_1 = s_phi_in_0 * phimdy_in @ phim.predict.weight.data.T
        phimdxx_in_1 = s_phi_in_0 * phimdxx_in @ phim.predict.weight.data.T
        phimdyy_in_1 = s_phi_in_0 * phimdyy_in @ phim.predict.weight.data.T

        A1 = (c1mdt_in - D1 * (c1mdxx_in + c1mdyy_in) - D1 * z1 * (
                c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1 + c1m_in * (
                phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
        b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt_in - D2 * (c2mdxx_in + c2mdyy_in) - D2 * z2 * (
                c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1 + c2m_in * (
                phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
        b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        A3 = (c3mdt_in - D2 * (c3mdxx_in + c3mdyy_in) - D2 * z2 * (
                c3mdx_in * phimdx_in_1 + c3mdy_in * phimdy_in_1 + c3m_in * (
                phimdxx_in_1 + phimdyy_in_1))).detach().cpu().numpy()
        B3 = (c3mdx_b * n1 + c3mdy_b * n2).detach().cpu().numpy()
        I3 = (c3m_i).detach().cpu().numpy()

        A_NP3 = np.vstack((A3, p1 * B3, p1 * I3)) 
        b_NP3 = np.vstack((fa3, p1 * fb3, p1 * fi3))
        wu_NP3 = np.linalg.lstsq(A_NP3, b_NP3)[0] 
        c3m.predict.weight.data = torch.from_numpy(wu_NP3[0:m]).T.to(device)

        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)
        mass3_re = torch.zeros(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)
                mass3_re[i] = err_calculate.int_2d(c3m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re
        s_c3 = mass3 / mass3_re

        xt_re_np = xt_re.squeeze().numpy()
        s_c1_np = s_c1.squeeze().detach().numpy()
        s_c2_np = s_c2.squeeze().detach().numpy()
        s_c3_np = s_c3.squeeze().detach().numpy()

        s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
        s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
        s_c3_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c3_np, kind='linear', fill_value='extrapolate')
        s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        s_c3_in_0 = torch.from_numpy(s_c3_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

        # P
        c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
        c2m_in_1 = s_c1_in_0 * c2m(xy_inter)[1]
        c3m_in_1 = s_c1_in_0 * c3m(xy_inter)[1]

        A4 = -eps ** 2 * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
        fa4 = (z1 * c1m_in_1 + z2 * c2m_in_1 + z3 * c3m_in_1).detach().cpu().numpy()
        B4 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

        A_P = np.vstack((A4, p1 * B4, E1)) 
        b_P = np.vstack((fa4, p1 * fb4, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        end3 = time.perf_counter()
        total_time3 += (end3 - start3)

        # s_phi_np = s_phi.squeeze().detach().numpy()

        # s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
        # s_phi_int_0 = torch.from_numpy(s_phi_np)

        # s_phi_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='cubic', fill_value='extrapolate')
        # s_phi_int_0 = s_phi_cubic_0(xt_re.squeeze().numpy())

        for i in range(Nre):
            x2_plot = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            c1m_plot = s_c1[i] * c1m(x2_plot)[1]
            c2m_plot = s_c2[i] * c2m(x2_plot)[1]
            c3m_plot = s_c3[i] * c3m(x2_plot)[1]
            phid1_plot = phim(x2_plot)[2]
            results['c1_min'][i] = torch.min(c1m_plot)
            results['c2_min'][i] = torch.min(c2m_plot)
            results['c3_min'][i] = torch.min(c3m_plot)
            results['m1'][i] = err_calculate.int_2d(s_c1[i] * c1m(x2_plot)[1], wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(s_c2[i] * c2m(x2_plot)[1], wr2, a, b, c, d)
            results['m3'][i] = err_calculate.int_2d(s_c3[i] * c3m(x2_plot)[1], wr2, a, b, c, d)
            psidx_plot = ((phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_plot) @ phim.predict.weight.data.T)
            psidy_plot = ((phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_plot) @ phim.predict.weight.data.T)
            energy_terms = c1m_plot * torch.log(c1m_plot) + c2m_plot * torch.log(c2m_plot) + c3m_plot * torch.log(
                c3m_plot) + eps ** 2 / 2 * (psidx_plot ** 2 + psidy_plot ** 2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

        xt_re_all[(k + 1) * Nre:(k + 2) * Nre] = xt_re
        results_all['e_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e']
        results_all['e_bar_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e_bar']
        results_all['m1_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1']
        results_all['m2_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2']
        results_all['m3_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m3']
        results_all['c1_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_min']
        results_all['c2_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_min']
        results_all['c3_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c3_min']

        PNP_plot_st_re_G_benchmark(xt_re, results, k + 1)

        t1 = t2

    PNP_plot_st_re_G_all_benchmark(xt_re_all, results_all)

    print(total_time1)
    print(total_time2)
    print(total_time3)
    total_time = total_time1 + total_time2 + total_time3
    print(total_time)
