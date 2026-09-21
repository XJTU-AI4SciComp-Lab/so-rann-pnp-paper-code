from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.experiment_io import legacy_output_path

import ex1_real
import numpy as np
from RNN_ref import err_calculate
from RNN_ref import fd
import torch
from RNN_ref import sampling_points
import time
from RNN_ref import RNN_funs
import scipy

torch.set_default_dtype(torch.float64)


def _refit_phi_after_mass_correction(
    phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
    A_P, fb3, fe1, p1, eps, z1, z2, m, device,
):
    """Fit the reported potential to the final mass-corrected charge density."""
    c1_cut_in = c1m(xy_inter)[1]
    c2_cut_in = c2m(xy_inter)[1]
    time_nodes = xt_re.detach().cpu().numpy()
    interior_times = xy_inter[:, 2].detach().cpu().numpy()
    scale1_in = torch.as_tensor(
        np.interp(interior_times, time_nodes, s_c1.detach().cpu().numpy()),
        dtype=c1_cut_in.dtype,
        device=c1_cut_in.device,
    ).reshape(-1, 1)
    scale2_in = torch.as_tensor(
        np.interp(interior_times, time_nodes, s_c2.detach().cpu().numpy()),
        dtype=c2_cut_in.dtype,
        device=c2_cut_in.device,
    ).reshape(-1, 1)
    rhs_in = (
        ex1_real.f3(xy_inter, eps, z1, z2)
        + z1 * scale1_in * c1_cut_in
        + z2 * scale2_in * c2_cut_in
    ).detach().cpu().numpy()
    rhs = np.vstack((rhs_in, p1 * fb3, fe1))
    weights = np.linalg.lstsq(A_P, rhs, rcond=None)[0][:m]
    phim.predict.weight.data = torch.from_numpy(weights).T.to(device)


def PNPNS_plot_st_re(xt_re, results, k):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flat

    plots_config = [
        (ax1, 'c1_error', 'c1_error'),
        (ax2, 'c2_error', 'c2_error'),
        (ax3, 'phi_error', 'phi_error'),
        (ax4, 'u1_error', 'u1_error'),
        (ax5, 'u2_error', 'u2_error'),
        (ax6, 'p_error', 'p_error'),
        (ax7, 'c1_min', 'c1_min'),
        (ax8, 'c2_min', 'c2_min'),
        (ax9, 'm1', 'mass1'),
        (ax10, 'm2', 'mass2'),
        (ax11, 'Divergence_free', 'L2_nrom')
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    ax12.plot(xt_re, results['e'], label='numerical')
    ax12.plot(xt_re, results['e_real'], linestyle='--', label='real')
    ax12.set(xlabel='t', ylabel='e', title='energy')
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex1/PNPNS-st-re-ex1_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNPNS_plot_st_re_all(xt_re, results):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flat

    plots_config = [
        (ax1, 'c1_error_all', 'c1_error'),
        (ax2, 'c2_error_all', 'c2_error'),
        (ax3, 'phi_error_all', 'phi_error'),
        (ax4, 'u1_error_all', 'u1_error'),
        (ax5, 'u2_error_all', 'u2_error'),
        (ax6, 'p_error_all', 'p_error'),
        (ax7, 'c1_min_all', 'c1_min'),
        (ax8, 'c2_min_all', 'c2_min'),
        (ax9, 'm1_all', 'mass1'),
        (ax10, 'm2_all', 'mass2'),
        (ax11, 'Divergence_free_all', 'L2_nrom'),
        (ax12, 'e_all', 'energy_pred')
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex1/PNPNS-st-re-ex1-all.eps'), bbox_inches='tight')
    plt.close(fig)

def PNPNS_plot_st(xt_re, results, k):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flat

    plots_config = [
        (ax1, 'c1_error', 'c1_error'),
        (ax2, 'c2_error', 'c2_error'),
        (ax3, 'phi_error', 'phi_error'),
        (ax4, 'u1_error', 'u1_error'),
        (ax5, 'u2_error', 'u2_error'),
        (ax6, 'p_error', 'p_error'),
        (ax7, 'c1_min', 'c1_min'),
        (ax8, 'c2_min', 'c2_min'),
        (ax9, 'm1', 'mass1'),
        (ax10, 'm2', 'mass2'),
        (ax11, 'Divergence_free', 'L2_nrom')
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    ax12.plot(xt_re, results['e'], label='numerical')
    ax12.plot(xt_re, results['e_real'], linestyle='--', label='real')
    ax12.set(xlabel='t', ylabel='e', title='energy')
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex1/PNPNS-st-ex1_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNPNS_plot_st_all(xt_re, results):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flat

    plots_config = [
        (ax1, 'c1_error_all', 'c1_error'),
        (ax2, 'c2_error_all', 'c2_error'),
        (ax3, 'phi_error_all', 'phi_error'),
        (ax4, 'u1_error_all', 'u1_error'),
        (ax5, 'u2_error_all', 'u2_error'),
        (ax6, 'p_error_all', 'p_error'),
        (ax7, 'c1_min_all', 'c1_min'),
        (ax8, 'c2_min_all', 'c2_min'),
        (ax9, 'm1_all', 'mass1'),
        (ax10, 'm2_all', 'mass2'),
        (ax11, 'Divergence_free_all', 'L2_nrom'),
        (ax12, 'e_all', 'energy_pred')
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex1/PNPNS-st-ex1-all.eps'), bbox_inches='tight')
    plt.close(fig)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def pnpns_ex1_picard_st_decoupled_remass(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, ite, p1):
    device = 'cpu'
    total_time1 = 0        
    total_time2 = 0        
    total_time3 = 0        
    picard_iterations = []

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre),
        'm1_raw': np.zeros(Nre), 'm2_raw': np.zeros(Nre),
        'm1_cut': np.zeros(Nre), 'm2_cut': np.zeros(Nre),
        'e': np.zeros(Nre), 'Divergence_free': np.zeros(Nre),
        'e_real': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre),
        'c1_error': np.zeros(Nre), 'c2_error': np.zeros(Nre), 'phi_error': np.zeros(Nre),
        'u1_error': np.zeros(Nre), 'u2_error': np.zeros(Nre), 'p_error': np.zeros(Nre),
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt),
        'm1_raw_all': np.zeros(Nre * Nt), 'm2_raw_all': np.zeros(Nre * Nt),
        'm1_cut_all': np.zeros(Nre * Nt), 'm2_cut_all': np.zeros(Nre * Nt),
        'Divergence_free_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt),
        'c1_error_all': np.zeros(Nre * Nt), 'c2_error_all': np.zeros(Nre * Nt), 'phi_error_all': np.zeros(Nre * Nt),
        'u1_error_all': np.zeros(Nre * Nt), 'u2_error_all': np.zeros(Nre * Nt), 'p_error_all': np.zeros(Nre * Nt)
    }

    scaler = {
        'scaler_c1': torch.zeros(Nre * Nt), 'scaler_c2': torch.zeros(Nre * Nt),
    }

    start1 = time.perf_counter()

    xr = sampling_points.get_GL_points_3d(n_int)
    wr = sampling_points.get_weights_3d(n_int).reshape([n_int ** 3, 1])
    x = torch.zeros_like(xr)
    x[:, 0:1] = (b - a) / 2 * xr[:, 0:1] + (a + b) / 2
    x[:, 1:2] = (d - c) / 2 * xr[:, 1:2] + (c + d) / 2

    x2_0 = torch.cat((x2, torch.zeros(x2.shape[0], 1)), dim=1)
    with torch.no_grad():
        mass1 = err_calculate.int_2d(ex1_real.c1(x2_0), wr2, a, b, c, d)
        mass2 = err_calculate.int_2d(ex1_real.c2(x2_0), wr2, a, b, c, d)

    t2 = t1 + s
    RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(um, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(pm, m, a, b, c, d, t1, t2)

    N1 = int((d-c)*(b-a)*s*n**3)
    N2x = int((b-a)*s*n**2)
    N2y = int((d-c)*s*n**2)
    N3 = int((d-c)*(b-a)*n**2)
    xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
    xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
    xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
    xt_re = torch.linspace(t1, t2, Nre)
    x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

    nor = torch.zeros(2*(N2x+N2y), 2)
    nor[0:N2x, 1:2] = -1
    nor[N2x:2*N2x, 1:2] = 1
    nor[2*N2x:2*N2x+N2y, 0:1] = -1
    nor[2*N2x+N2y:2*(N2x+N2y), 0:1] = 1

    n1, n2 = nor[:, 0:1], nor[:, 1:2]

    N_in = xy_inter.shape[0]
    N_b = xb.shape[0]
    N_i = xi.shape[0]
    C1 = np.zeros((N_in, 2*m))           
    C2 = np.zeros((N_in, 2*m))
    B4 = np.zeros((N_b, 2*m))             #u1
    B5 = np.zeros((N_b, 2*m))             #u2
    I3 = np.zeros((N_i, 2*m))             #u1
    I4 = np.zeros((N_i, 2*m))             #u2

    fa1 = (ex1_real.f1(xy_inter, z1, D1)).detach().cpu().numpy()
    fa2 = (ex1_real.f2(xy_inter, z2, D2)).detach().cpu().numpy()
    fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).detach().cpu().numpy()
    fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).detach().cpu().numpy()
    fb3 = (ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2).detach().cpu().numpy()
    fb4 = (ex1_real.u1(xb)).detach().cpu().numpy()
    fb5 = (ex1_real.u2(xb)).detach().cpu().numpy()
    fi1 = (ex1_real.c1(xi)).detach().cpu().numpy()
    fi2 = (ex1_real.c2(xi)).detach().cpu().numpy()
    fi3 = (ex1_real.u1(xi)).detach().cpu().numpy()
    fi4 = (ex1_real.u2(xi)).detach().cpu().numpy()

    # t = torch.linspace(t1, t2, n).reshape(-1, 1)
    # E1 = np.zeros((n, m))
    # fe1 = np.zeros((n, 1))
    # E2 = np.zeros((n, 2*m))
    # fe2 = np.zeros((n, 1))
    # scale_factor = (b - a) * (d - c) / 4
    # for j in range(n):
    #     x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
    #     with torch.no_grad():
    #         output_p = torch.zeros(n_int ** 2, 2 * m)
    #         output_phi = phim(x2_t)[0]
    #         exact_phi = ex1_real.phi(x2_t)
    #         output_p[:, m:2 * m] = pm(x2_t)[0]
    #         exact_p = ex1_real.p(x2_t)
    #         E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
    #         fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()
    #         E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0, keepdim=True)).detach().cpu().numpy()
    #         fe2[j:j + 1, :] = (scale_factor * torch.sum(exact_p * wr2, dim=0)).detach().cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    E2 = np.zeros((n, 2*m))
    fe2 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_p = torch.zeros(n_int ** 2, 2 * m)
            output_phi = phim(x2_t)[0]
            output_p[:, m:2 * m] = pm(x2_t)[0]
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
            E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0, keepdim=True)).detach().cpu().numpy()

    c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
    c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
    phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

    umd1_in = um(xy_inter)[5]
    umd2_in = um(xy_inter)[6]
    u1mdx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u1mdy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd1_in
    u1mdt_in = um.hidden.weight.data[:, 2:3].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u2mdx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * umd1_in
    u2mdy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u2mdt_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 2:3].reshape([-1, m]) * umd1_in
    u1mdxx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd2_in
    u1mdyy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 3 * umd2_in
    u2mdxx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 3 * umd2_in
    u2mdyy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd2_in

    pmd1_in = pm(xy_inter)[2]
    pmdx_in = pm.hidden.weight.data[:, 0:1].reshape([-1, m]) * pmd1_in
    pmdy_in = pm.hidden.weight.data[:, 1:2].reshape([-1, m]) * pmd1_in

    c1md1_b = c1m(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2md1_b = c2m(xb)[2]
    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    phimd1_b = phim(xb)[2]
    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    u1m_b = um(xb)[1]
    u2m_b = um(xb)[2]

    c1m_i = c1m(xi)[0]
    c2m_i = c2m(xi)[0]
    u1m_i = um(xi)[1]
    u2m_i = um(xi)[2]

    end1 = time.perf_counter()
    total_time1 += (end1 - start1)

    diffc1, diffc2, diffphi, diffu1, diffu2, diffp = 1, 1, 1, 1, 1, 1
    ttt = 0

    while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold or diffu1 > threshold or diffu2 > threshold or diffp > threshold) and ttt<ite:
        start2 = time.perf_counter()
        c1_before, c2_before, phi_before, u1_before, u2_before, p_before= c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]

        # NP
        u1m_in_1 = um(xy_inter)[3]
        u2m_in_1 = um(xy_inter)[4]
        phimdx_in_1 = phimdx_in@phim.predict.weight.data.T
        phimdy_in_1 = phimdy_in@phim.predict.weight.data.T
        phimdxx_in_1 = phimdxx_in@phim.predict.weight.data.T
        phimdyy_in_1 = phimdyy_in@phim.predict.weight.data.T

        A1 = (c1mdt_in + (u1m_in_1*c1mdx_in + u2m_in_1*c1mdy_in) - D1*(c1mdxx_in + c1mdyy_in + z1*(c1m_in*(phimdxx_in_1 + phimdyy_in_1) + c1mdx_in*phimdx_in_1 + c1mdy_in*phimdy_in_1))).detach().cpu().numpy()
        B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
        b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt_in + (u1m_in_1*c2mdx_in + u2m_in_1*c2mdy_in) - D2*(c2mdxx_in + c2mdyy_in + z2*(c2m_in*(phimdxx_in_1 + phimdyy_in_1) + c2mdx_in*phimdx_in_1 + c2mdy_in*phimdy_in_1))).detach().cpu().numpy()
        B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
        b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        # P
        c1m_in_1 = c1m(xy_inter)[1]
        c2m_in_1 = c2m(xy_inter)[1]

        A3 = -eps*(phimdxx_in + phimdyy_in).detach().cpu().numpy()
        fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + (z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1*B3, E1)) 
        b_P = np.vstack((fa3, p1*fb3, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        # NS
        phimdx_1 = phimdx_in@phim.predict.weight.data.T
        phimdy_1 = phimdy_in@phim.predict.weight.data.T

        C1[:, 0:m] = (u1mdt_in + u1m_in_1*u1mdx_in + u2m_in_1*u1mdy_in - nv*(u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
        C1[:, m:2*m] = (pmdx_in).detach().cpu().numpy()
        fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
        C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
        fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
        B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
        I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
        I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

        A_NS = np.vstack((C1, C2, p1*B4, p1*B5, p1*I3, p1*I4, E2)) 
        b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, p1*fi3, p1*fi4, fe2))
        # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
        wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
        um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
        pm.predict.weight.data = torch.from_numpy(wu_NS[m:2*m]).T.to(device)

        c1_after, c2_after, phi_after, u1_after, u2_after, p_after = c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]
        diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
        diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
        diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
        diffu1 = err_calculate.err_L2_3d(u1_after, u1_before, wr, a, b, c, d, t1, t2)
        diffu2 = err_calculate.err_L2_3d(u2_after, u2_before, wr, a, b, c, d, t1, t2)
        diffp = err_calculate.err_L2_3d(p_after, p_before, wr, a, b, c, d, t1, t2)

        ttt = ttt + 1

        end2 = time.perf_counter()
        total_time2 += (end2 - start2)
    print(ttt)
    picard_iterations.append(int(ttt))

    start3 = time.perf_counter()

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re

    _refit_phi_after_mass_correction(
        phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
        A_P, fb3, fe1, p1, eps, z1, z2, m, device,
    )

    end3 = time.perf_counter()
    total_time3 += (end3 - start3)

    # # s_c1 = torch.ones(Nre)
    # # s_c2 = torch.ones(Nre)
    #
    # xt_re_np = xt_re.squeeze().numpy()
    # s_c1_np = s_c1.squeeze().detach().numpy()
    # s_c2_np = s_c2.squeeze().detach().numpy()
    #
    # s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
    # s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
    # # s_c1_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='cubic', fill_value='extrapolate')
    # # s_c2_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='cubic', fill_value='extrapolate')
    #
    # s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    # s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    #
    # # s_c1_in_0 = s_c1_cubic_0(xy_inter[:, 2:3].squeeze().numpy())
    # # s_c2_in_0 = s_c2_cubic_0(xy_inter[:, 2:3].squeeze().numpy())
    #
    # c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
    # c2m_in_1 = s_c2_in_0 * c2m(xy_inter)[1]
    #
    # # P
    # A3 = -eps * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
    # fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1 * c1m_in_1 + z2 * c2m_in_1).detach().cpu().numpy()
    # B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()
    #
    # b_P = np.vstack((fa3, p1 * fb3, fe1))
    # phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

    # # NS
    # u1m_in_1 = um(xy_inter)[3]
    # u2m_in_1 = um(xy_inter)[4]
    # phimdx_1 = phimdx_in @ phim.predict.weight.data.T
    # phimdy_1 = phimdy_in @ phim.predict.weight.data.T
    #
    # C1[:, 0:m] = (u1mdt_in + u1m_in_1 * u1mdx_in + u2m_in_1 * u1mdy_in - nv * (u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
    # C1[:, m:2 * m] = (pmdx_in).detach().cpu().numpy()
    # fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()
    #
    # C2[:, 0:m] = (u2mdt_in + u1m_in_1 * u2mdx_in + u2m_in_1 * u2mdy_in - nv * (u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
    # C2[:, m:2 * m] = (pmdy_in).detach().cpu().numpy()
    # fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()
    #
    # B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
    # B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
    # I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
    # I4[:, 0:m] = (u2m_i).detach().cpu().numpy()
    #
    # b_NS = np.vstack((fc1, fc2, p1 * fb4, p1 * fb5, p1 * fi3, p1 * fi4, fe2))
    # # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
    # um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
    # pm.predict.weight.data = torch.from_numpy(wu_NS[m:2 * m]).T.to(device)

    # u1mdx_in_1 = u1mdx_in @ um.predict.weight.data.T
    # u1mdy_in_1 = u1mdy_in @ um.predict.weight.data.T
    # u1mdt_in_1 = u1mdt_in @ um.predict.weight.data.T
    # u1mdxx_in_1 = u1mdxx_in @ um.predict.weight.data.T
    # u1mdyy_in_1 = u1mdyy_in @ um.predict.weight.data.T
    # u2mdx_in_1 = u2mdx_in @ um.predict.weight.data.T
    # u2mdy_in_1 = u2mdy_in @ um.predict.weight.data.T
    # u2mdt_in_1 = u2mdt_in @ um.predict.weight.data.T
    # u2mdxx_in_1 = u2mdxx_in @ um.predict.weight.data.T
    # u2mdyy_in_1 = u2mdyy_in @ um.predict.weight.data.T
    #
    # E3 = E2[:, m:2*m]
    #
    # C3 = pmdx_in.detach().cpu().numpy()
    # fc3 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1 * (z1 * c1m_in_1 + z2 * c2m_in_1) - (u1mdt_in_1 + u1m_in_1 * u1mdx_in_1 + u2m_in_1 * u1mdy_in_1 - nv * (u1mdxx_in_1 + u1mdyy_in_1))).detach().cpu().numpy()
    # C4 = pmdy_in.detach().cpu().numpy()
    # fc4 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1 * (z1 * c1m_in_1 + z2 * c2m_in_1) - (u2mdt_in_1 + u1m_in_1 * u2mdx_in_1 + u2m_in_1 * u2mdy_in_1 - nv * (u2mdxx_in_1 + u2mdyy_in_1))).detach().cpu().numpy()
    # b_pre = np.vstack((fc3, fc4, fe2))
    # pm.predict.weight.data = torch.from_numpy(wu_pre[0:m]).T.to(device)

    # s_c1_int_0 = torch.from_numpy(s_c1_linear_0(xt_re.squeeze().numpy()))
    # s_c2_int_0 = torch.from_numpy(s_c2_linear_0(xt_re.squeeze().numpy()))

    # s_c1_int_0 = s_c1_cubic_0(xt_re.squeeze().numpy())
    # s_c2_int_0 = s_c2_cubic_0(xt_re.squeeze().numpy())

    torch.save(c1m, legacy_output_path('PNPNS_code/model_ex1/c1m_0.pth'))
    torch.save(c2m, legacy_output_path('PNPNS_code/model_ex1/c2m_0.pth'))
    torch.save(phim, legacy_output_path('PNPNS_code/model_ex1/phim_0.pth'))
    # torch.save(um, legacy_output_path('PNPNS_code/model_ex1/um_0.pth'))
    torch.save(um.state_dict(), legacy_output_path('PNPNS_code/model_ex1/um_weights_0.pth'))
    torch.save(pm, legacy_output_path('PNPNS_code/model_ex1/pm_0.pth'))
    scaler['scaler_c1'][0:Nre] = s_c1
    scaler['scaler_c2'][0:Nre] = s_c2

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        x2_re.requires_grad_(True)
        c1_hidden, c1m_plot_1 = c1m(x2_re)[0], c1m(x2_re)[1]
        c2_hidden, c2m_plot_1 = c2m(x2_re)[0], c2m(x2_re)[1]
        c1_raw = c1m.predict(c1_hidden)
        c2_raw = c2m.predict(c2_hidden)
        phim_plot_1 = phim(x2_re)[1]
        phimd1_plot = phim(x2_re)[2]
        u1m_plot_1 = um(x2_re)[3]
        u2m_plot_1 = um(x2_re)[4]
        pm_plot_1 = pm(x2_re)[1]
        results['c1_error'][i] = err_calculate.err_L2_2d(s_c1[i]*c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
        results['c2_error'][i] = err_calculate.err_L2_2d(s_c2[i]*c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
        results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
        results['u1_error'][i] = err_calculate.err_L2_2d(u1m_plot_1, ex1_real.u1(x2_re), wr2, a, b, c, d)
        results['u2_error'][i] = err_calculate.err_L2_2d(u2m_plot_1, ex1_real.u2(x2_re), wr2, a, b, c, d)
        results['p_error'][i] = err_calculate.err_L2_2d(pm_plot_1, ex1_real.p(x2_re), wr2, a, b, c, d)
        results['c1_min'][i] = torch.min(s_c1[i] * c1m_plot_1)
        results['c2_min'][i] = torch.min(s_c2[i] * c2m_plot_1)
        results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m_plot_1, wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m_plot_1, wr2, a, b, c, d)
        results['m1_raw'][i] = err_calculate.int_2d(c1_raw, wr2, a, b, c, d)
        results['m2_raw'][i] = err_calculate.int_2d(c2_raw, wr2, a, b, c, d)
        results['m1_cut'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
        results['m2_cut'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
        results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot_1, u2m_plot_1, x2_re, wr2, a, b, c, d)
        phidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        phidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        energy_terms = s_c1[i]*c1m_plot_1 * torch.log(s_c1[i]*c1m_plot_1) + s_c2[i]*c2m_plot_1 * torch.log(s_c2[i]*c2m_plot_1) + eps ** 2 / 2 * (phidx ** 2 + phidy ** 2) + 0.5*(u1m_plot_1**2 + u2m_plot_1**2)
        real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps**2/2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2) + 0.5*(ex1_real.u1(x2_re)**2 + ex1_real.u2(x2_re)**2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
        results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

    print(f'c1_L^2 error: {results["c1_error"][-1]}')
    print(f'c2_L^2 error: {results["c2_error"][-1]}')
    print(f'phi_L^2 error: {results["phi_error"][-1]}')
    print(f'u1_L^2 error: {results["u1_error"][-1]}')
    print(f'u2_L^2 error: {results["u2_error"][-1]}')
    print(f'p_L^2 error: {results["p_error"][-1]}')
    print(f'divergence-free: {results["Divergence_free"][-1]}')

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['m1_raw_all'][0:Nre] = results['m1_raw']
    results_all['m2_raw_all'][0:Nre] = results['m2_raw']
    results_all['m1_cut_all'][0:Nre] = results['m1_cut']
    results_all['m2_cut_all'][0:Nre] = results['m2_cut']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['c1_error_all'][0:Nre] = results['c1_error']
    results_all['c2_error_all'][0:Nre] = results['c2_error']
    results_all['phi_error_all'][0:Nre] = results['phi_error']
    results_all['u1_error_all'][0:Nre] = results['u1_error']
    results_all['u2_error_all'][0:Nre] = results['u2_error']
    results_all['p_error_all'][0:Nre] = results['p_error']
    results_all['Divergence_free_all'][0:Nre] = results['Divergence_free']

    PNPNS_plot_st_re(xt_re, results, 0)

    t1 = t2

    for k in range(Nt-1):

        start1 = time.perf_counter()

        t2 = t1 + s
        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        N_in = xy_inter.shape[0]
        N_b = xb.shape[0]
        N_i = xi.shape[0]
        C1 = np.zeros((N_in, 2 * m)) 
        C2 = np.zeros((N_in, 2 * m))
        B4 = np.zeros((N_b, 2 * m))   # u1
        B5 = np.zeros((N_b, 2 * m))   # u2
        I3 = np.zeros((N_i, 2 * m))   # u1
        I4 = np.zeros((N_i, 2 * m))   # u2

        fa1 = (ex1_real.f1(xy_inter, z1, D1)).detach().cpu().numpy()
        fa2 = (ex1_real.f2(xy_inter, z2, D2)).detach().cpu().numpy()
        fb1 = (ex1_real.c1x(xb) * n1 + ex1_real.c1y(xb) * n2).detach().cpu().numpy()
        fb2 = (ex1_real.c2x(xb) * n1 + ex1_real.c2y(xb) * n2).detach().cpu().numpy()
        fb3 = (ex1_real.phix(xb) * n1 + ex1_real.phiy(xb) * n2).detach().cpu().numpy()
        fb4 = (ex1_real.u1(xb)).detach().cpu().numpy()
        fb5 = (ex1_real.u2(xb)).detach().cpu().numpy()
        fi1 = (c1m(xi)[1]).detach().cpu().numpy()
        fi2 = (c2m(xi)[1]).detach().cpu().numpy()
        fi3 = (um(xi)[3]).detach().cpu().numpy()
        fi4 = (um(xi)[4]).detach().cpu().numpy()

        RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(um, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(pm, m, a, b, c, d, t1, t2)

        # t = torch.linspace(t1, t2, n).reshape(-1, 1)
        # E1 = np.zeros((n, m))
        # fe1 = np.zeros((n, 1))
        # E2 = np.zeros((n, 2 * m))
        # fe2 = np.zeros((n, 1))
        # scale_factor = (b - a) * (d - c) / 4
        # for j in range(n):
        #     x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
        #     with torch.no_grad():
        #         output_p = torch.zeros(n_int ** 2, 2 * m)
        #         output_phi = phim(x2_t)[0]
        #         exact_phi = ex1_real.phi(x2_t)
        #         output_p[:, m:2 * m] = pm(x2_t)[0]
        #         exact_p = ex1_real.p(x2_t)
        #         E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
        #         fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()
        #         E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0,keepdim=True)).detach().cpu().numpy()
        #         fe2[j:j + 1, :] = (scale_factor * torch.sum(exact_p * wr2, dim=0)).detach().cpu().numpy()

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        E2 = np.zeros((n, 2 * m))
        fe2 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_p = torch.zeros(n_int ** 2, 2 * m)
                output_phi = phim(x2_t)[0]
                output_p[:, m:2 * m] = pm(x2_t)[0]
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
                E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0, keepdim=True)).detach().cpu().numpy()

        c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
        c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
        phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

        umd1_in = um(xy_inter)[5]
        umd2_in = um(xy_inter)[6]
        u1mdx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u1mdy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd1_in
        u1mdt_in = um.hidden.weight.data[:, 2:3].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u2mdx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * umd1_in
        u2mdy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u2mdt_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 2:3].reshape([-1, m]) * umd1_in
        u1mdxx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd2_in
        u1mdyy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 3 * umd2_in
        u2mdxx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 3 * umd2_in
        u2mdyy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd2_in

        pmd1_in = pm(xy_inter)[2]
        pmdx_in = pm.hidden.weight.data[:, 0:1].reshape([-1, m]) * pmd1_in
        pmdy_in = pm.hidden.weight.data[:, 1:2].reshape([-1, m]) * pmd1_in

        c1md1_b = c1m(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2md1_b = c2m(xb)[2]
        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        phimd1_b = phim(xb)[2]
        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        u1m_b = um(xb)[1]
        u2m_b = um(xb)[2]

        c1m_i = c1m(xi)[0]
        c2m_i = c2m(xi)[0]
        u1m_i = um(xi)[1]
        u2m_i = um(xi)[2]

        end1 = time.perf_counter()
        total_time1 += (end1 - start1)

        diffc1, diffc2, diffphi, diffu1, diffu2, diffp = 1, 1, 1, 1, 1, 1
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold or diffu1 > threshold or diffu2 > threshold or diffp > threshold) and ttt<ite:
            start2 = time.perf_counter()
            c1_before, c2_before, phi_before, u1_before, u2_before, p_before= c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]

            # NP
            u1m_in_1 = um(xy_inter)[3]
            u2m_in_1 = um(xy_inter)[4]
            phimdx_in_1 = phimdx_in@phim.predict.weight.data.T
            phimdy_in_1 = phimdy_in@phim.predict.weight.data.T
            phimdxx_in_1 = phimdxx_in@phim.predict.weight.data.T
            phimdyy_in_1 = phimdyy_in@phim.predict.weight.data.T

            A1 = (c1mdt_in + (u1m_in_1*c1mdx_in + u2m_in_1*c1mdy_in) - D1*(c1mdxx_in + c1mdyy_in + z1*(c1m_in*(phimdxx_in_1 + phimdyy_in_1) + c1mdx_in*phimdx_in_1 + c1mdy_in*phimdy_in_1))).detach().cpu().numpy()
            B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
            b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt_in + (u1m_in_1*c2mdx_in + u2m_in_1*c2mdy_in) - D2*(c2mdxx_in + c2mdyy_in + z2*(c2m_in*(phimdxx_in_1 + phimdyy_in_1) + c2mdx_in*phimdx_in_1 + c2mdy_in*phimdy_in_1))).detach().cpu().numpy()
            B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
            b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            # P
            c1m_in_1 = c1m(xy_inter)[1]
            c2m_in_1 = c2m(xy_inter)[1]

            A3 = -eps*(phimdxx_in + phimdyy_in).detach().cpu().numpy()
            fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + (z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

            A_P = np.vstack((A3, p1*B3, E1)) 
            b_P = np.vstack((fa3, p1*fb3, fe1))
            wu_P = np.linalg.lstsq(A_P, b_P)[0] 
            phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

            # NS
            phimdx_1 = phimdx_in@phim.predict.weight.data.T
            phimdy_1 = phimdy_in@phim.predict.weight.data.T

            C1[:, 0:m] = (u1mdt_in + u1m_in_1*u1mdx_in + u2m_in_1*u1mdy_in - nv*(u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
            C1[:, m:2*m] = (pmdx_in).detach().cpu().numpy()
            fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
            C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
            fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
            B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
            I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
            I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

            A_NS = np.vstack((C1, C2, p1*B4, p1*B5, p1*I3, p1*I4, E2)) 
            b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, p1*fi3, p1*fi4, fe2))
            # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
            wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
            um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
            pm.predict.weight.data = torch.from_numpy(wu_NS[m:2*m]).T.to(device)

            c1_after, c2_after, phi_after, u1_after, u2_after, p_after = c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]
            diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
            diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
            diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
            diffu1 = err_calculate.err_L2_3d(u1_after, u1_before, wr, a, b, c, d, t1, t2)
            diffu2 = err_calculate.err_L2_3d(u2_after, u2_before, wr, a, b, c, d, t1, t2)
            diffp = err_calculate.err_L2_3d(p_after, p_before, wr, a, b, c, d, t1, t2)

            ttt = ttt + 1

            end2 = time.perf_counter()
            total_time2 += (end2 - start2)

        print(ttt)
        picard_iterations.append(int(ttt))
        start3 = time.perf_counter()
        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re

        _refit_phi_after_mass_correction(
            phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
            A_P, fb3, fe1, p1, eps, z1, z2, m, device,
        )

        end3 = time.perf_counter()
        total_time3 += (end3 - start3)

        # # s_c1 = torch.ones(Nre)
        # # s_c2 = torch.ones(Nre)
        #
        # xt_re_np = xt_re.squeeze().numpy()
        # s_c1_np = s_c1.squeeze().detach().numpy()
        # s_c2_np = s_c2.squeeze().detach().numpy()
        #
        # s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
        # s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
        # # s_c1_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='cubic', fill_value='extrapolate')
        # # s_c2_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='cubic', fill_value='extrapolate')
        #
        # s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        # s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        #
        # # s_c1_in_0 = s_c1_cubic_0(xy_inter[:, 2:3].squeeze().numpy())
        # # s_c2_in_0 = s_c2_cubic_0(xy_inter[:, 2:3].squeeze().numpy())
        #
        # c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
        # c2m_in_1 = s_c2_in_0 * c2m(xy_inter)[1]
        #
        # # P
        # A3 = -eps * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
        # fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1 * c1m_in_1 + z2 * c2m_in_1).detach().cpu().numpy()
        # B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()
        #
        # b_P = np.vstack((fa3, p1 * fb3, fe1))
        # phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)
        #
        # # NS
        # u1m_in_1 = um(xy_inter)[3]
        # u2m_in_1 = um(xy_inter)[4]
        # phimdx_1 = phimdx_in @ phim.predict.weight.data.T
        # phimdy_1 = phimdy_in @ phim.predict.weight.data.T
        #
        # C1[:, 0:m] = (u1mdt_in + u1m_in_1 * u1mdx_in + u2m_in_1 * u1mdy_in - nv * (u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
        # C1[:, m:2 * m] = (pmdx_in).detach().cpu().numpy()
        # fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()
        #
        # C2[:, 0:m] = (u2mdt_in + u1m_in_1 * u2mdx_in + u2m_in_1 * u2mdy_in - nv * (u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
        # C2[:, m:2 * m] = (pmdy_in).detach().cpu().numpy()
        # fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()
        #
        # B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
        # B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
        # I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
        # I4[:, 0:m] = (u2m_i).detach().cpu().numpy()
        #
        # b_NS = np.vstack((fc1, fc2, p1 * fb4, p1 * fb5, p1 * fi3, p1 * fi4, fe2))
        # # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
        # um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
        # pm.predict.weight.data = torch.from_numpy(wu_NS[m:2 * m]).T.to(device)

        # s_c1_int_0 = torch.from_numpy(s_c1_linear_0(xt_re.squeeze().numpy()))
        # s_c2_int_0 = torch.from_numpy(s_c2_linear_0(xt_re.squeeze().numpy()))

        # s_c1_int_0 = s_c1_cubic_0(xt_re.squeeze().numpy())
        # s_c2_int_0 = s_c2_cubic_0(xt_re.squeeze().numpy())

        torch.save(c1m, legacy_output_path(f'PNPNS_code/model_ex1/c1m_{k+1}.pth'))
        torch.save(c2m, legacy_output_path(f'PNPNS_code/model_ex1/c2m_{k+1}.pth'))
        torch.save(phim, legacy_output_path(f'PNPNS_code/model_ex1/phim_{k+1}.pth'))
        # torch.save(um, legacy_output_path('PNPNS_code/model_ex1/um_0.pth'))
        torch.save(um.state_dict(), legacy_output_path(f'PNPNS_code/model_ex1/um_weights_{k+1}.pth'))
        torch.save(pm, legacy_output_path(f'PNPNS_code/model_ex1/pm_{k+1}.pth'))
        scaler['scaler_c1'][0:Nre] = s_c1
        scaler['scaler_c2'][0:Nre] = s_c2

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            x2_re.requires_grad_(True)
            c1_hidden, c1m_plot_1 = c1m(x2_re)[0], c1m(x2_re)[1]
            c2_hidden, c2m_plot_1 = c2m(x2_re)[0], c2m(x2_re)[1]
            c1_raw = c1m.predict(c1_hidden)
            c2_raw = c2m.predict(c2_hidden)
            phim_plot_1 = phim(x2_re)[1]
            phimd1_plot = phim(x2_re)[2]
            u1m_plot_1 = um(x2_re)[3]
            u2m_plot_1 = um(x2_re)[4]
            pm_plot_1 = pm(x2_re)[1]
            results['c1_error'][i] = err_calculate.err_L2_2d(s_c1[i]*c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
            results['c2_error'][i] = err_calculate.err_L2_2d(s_c2[i]*c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
            results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
            results['u1_error'][i] = err_calculate.err_L2_2d(u1m_plot_1, ex1_real.u1(x2_re), wr2, a, b, c, d)
            results['u2_error'][i] = err_calculate.err_L2_2d(u2m_plot_1, ex1_real.u2(x2_re), wr2, a, b, c, d)
            results['p_error'][i] = err_calculate.err_L2_2d(pm_plot_1, ex1_real.p(x2_re), wr2, a, b, c, d)
            results['c1_min'][i] = torch.min(s_c1[i] * c1m_plot_1)
            results['c2_min'][i] = torch.min(s_c2[i] * c2m_plot_1)
            results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m_plot_1, wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m_plot_1, wr2, a, b, c, d)
            results['m1_raw'][i] = err_calculate.int_2d(c1_raw, wr2, a, b, c, d)
            results['m2_raw'][i] = err_calculate.int_2d(c2_raw, wr2, a, b, c, d)
            results['m1_cut'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
            results['m2_cut'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
            results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot_1, u2m_plot_1, x2_re, wr2, a, b, c, d)
            phidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            phidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            energy_terms = s_c1[i]*c1m_plot_1 * torch.log(s_c1[i]*c1m_plot_1) + s_c2[i]*c2m_plot_1 * torch.log(s_c2[i]*c2m_plot_1) + eps ** 2 / 2 * (phidx ** 2 + phidy ** 2) + 0.5*(u1m_plot_1**2 + u2m_plot_1**2)
            real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps**2/2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2) + 0.5*(ex1_real.u1(x2_re)**2 + ex1_real.u2(x2_re)**2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
            results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

        print(f'c1_L^2 error: {results["c1_error"][-1]}')
        print(f'c2_L^2 error: {results["c2_error"][-1]}')
        print(f'phi_L^2 error: {results["phi_error"][-1]}')
        print(f'u1_L^2 error: {results["u1_error"][-1]}')
        print(f'u2_L^2 error: {results["u2_error"][-1]}')
        print(f'p_L^2 error: {results["p_error"][-1]}')
        print(f'divergence-free: {results["Divergence_free"][-1]}')

        PNPNS_plot_st_re(xt_re, results, k+1)

        xt_re_all[(k + 1) * Nre:(k + 2) * Nre] = xt_re
        results_all['e_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e']
        results_all['m1_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1']
        results_all['m2_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2']
        results_all['m1_raw_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1_raw']
        results_all['m2_raw_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2_raw']
        results_all['m1_cut_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1_cut']
        results_all['m2_cut_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2_cut']
        results_all['c1_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_min']
        results_all['c2_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_min']
        results_all['c1_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_error']
        results_all['c2_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_error']
        results_all['phi_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['phi_error']
        results_all['u1_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['u1_error']
        results_all['u2_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['u2_error']
        results_all['p_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['p_error']
        results_all['Divergence_free_all'][(k + 1) * Nre:(k + 2) * Nre] = results['Divergence_free']

        t1 = t2

    PNPNS_plot_st_re_all(xt_re_all, results_all)

    print(total_time1)
    print(total_time2)
    print(total_time3)
    total_time = total_time1 + total_time2 + total_time3
    print(total_time)
    return {
        'time': xt_re_all.detach().cpu().numpy(),
        'results': {key: np.asarray(value).copy() for key, value in results_all.items()},
        'scaler': {key: value.detach().cpu().numpy() for key, value in scaler.items()},
        'picard_iterations': picard_iterations,
        'potential_diagnostics': {'reported_phi_uses_final_charge': True},
        'timings': {
            'setup': float(total_time1),
            'picard': float(total_time2),
            'correction': float(total_time3),
            'total': float(total_time),
        },
    }

def pnpns_ex1_picard_st_decoupled(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, ite, p1):
    device = 'cpu'
    total_time = 0

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'e': np.zeros(Nre), 'Divergence_free': np.zeros(Nre),
        'e_real': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre),
        'c1_error': np.zeros(Nre), 'c2_error': np.zeros(Nre), 'phi_error': np.zeros(Nre),
        'u1_error': np.zeros(Nre), 'u2_error': np.zeros(Nre), 'p_error': np.zeros(Nre),
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt), 'Divergence_free_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt),
        'c1_error_all': np.zeros(Nre * Nt), 'c2_error_all': np.zeros(Nre * Nt), 'phi_error_all': np.zeros(Nre * Nt),
        'u1_error_all': np.zeros(Nre * Nt), 'u2_error_all': np.zeros(Nre * Nt), 'p_error_all': np.zeros(Nre * Nt)
    }


    start = time.perf_counter() 

    xr = sampling_points.get_GL_points_3d(n_int)
    wr = sampling_points.get_weights_3d(n_int).reshape([n_int ** 3, 1])
    x = torch.zeros_like(xr)
    x[:, 0:1] = (b - a) / 2 * xr[:, 0:1] + (a + b) / 2
    x[:, 1:2] = (d - c) / 2 * xr[:, 1:2] + (c + d) / 2

    # x2_0 = torch.cat((x2, torch.zeros(x2.shape[0], 1)), dim=1)
    # with torch.no_grad():
    #     mass1 = err_calculate.int_2d(ex1_real.c1(x2_0), wr2, a, b, c, d)
    #     mass2 = err_calculate.int_2d(ex1_real.c2(x2_0), wr2, a, b, c, d)

    t2 = t1 + s
    RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(um, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(pm, m, a, b, c, d, t1, t2)

    N1 = int((d-c)*(b-a)*s*n**3)
    N2x = int((b-a)*s*n**2)
    N2y = int((d-c)*s*n**2)
    N3 = int((d-c)*(b-a)*n**2)
    xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
    xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
    xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
    xt_re = torch.linspace(t1, t2, Nre)
    x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

    nor = torch.zeros(2*(N2x+N2y), 2)
    nor[0:N2x, 1:2] = -1
    nor[N2x:2*N2x, 1:2] = 1
    nor[2*N2x:2*N2x+N2y, 0:1] = -1
    nor[2*N2x+N2y:2*(N2x+N2y), 0:1] = 1

    n1, n2 = nor[:, 0:1], nor[:, 1:2]

    N_in = xy_inter.shape[0]
    N_b = xb.shape[0]
    N_i = xi.shape[0]
    C1 = np.zeros((N_in, 2*m))           
    C2 = np.zeros((N_in, 2*m))
    B4 = np.zeros((N_b, 2*m))             #u1
    B5 = np.zeros((N_b, 2*m))             #u2
    I3 = np.zeros((N_i, 2*m))             #u1
    I4 = np.zeros((N_i, 2*m))             #u2

    fa1 = (ex1_real.f1(xy_inter, z1, D1)).detach().cpu().numpy()
    fa2 = (ex1_real.f2(xy_inter, z2, D2)).detach().cpu().numpy()
    fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).detach().cpu().numpy()
    fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).detach().cpu().numpy()
    fb3 = (ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2).detach().cpu().numpy()
    fb4 = (ex1_real.u1(xb)).detach().cpu().numpy()
    fb5 = (ex1_real.u2(xb)).detach().cpu().numpy()
    fi1 = (ex1_real.c1(xi)).detach().cpu().numpy()
    fi2 = (ex1_real.c2(xi)).detach().cpu().numpy()
    fi3 = (ex1_real.u1(xi)).detach().cpu().numpy()
    fi4 = (ex1_real.u2(xi)).detach().cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    E2 = np.zeros((n, 2*m))
    fe2 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_p = torch.zeros(n_int ** 2, 2 * m)
            output_phi = phim(x2_t)[0]
            exact_phi = ex1_real.phi(x2_t)
            output_p[:, m:2 * m] = pm(x2_t)[0]
            exact_p = ex1_real.p(x2_t)
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
            fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()
            E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0, keepdim=True)).detach().cpu().numpy()
            fe2[j:j + 1, :] = (scale_factor * torch.sum(exact_p * wr2, dim=0)).detach().cpu().numpy()

    c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
    c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
    phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

    umd1_in = um(xy_inter)[5]
    umd2_in = um(xy_inter)[6]
    u1mdx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u1mdy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd1_in
    u1mdt_in = um.hidden.weight.data[:, 2:3].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u2mdx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * umd1_in
    u2mdy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
    u2mdt_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 2:3].reshape([-1, m]) * umd1_in
    u1mdxx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd2_in
    u1mdyy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 3 * umd2_in
    u2mdxx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 3 * umd2_in
    u2mdyy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd2_in

    pmd1_in = pm(xy_inter)[2]
    pmdx_in = pm.hidden.weight.data[:, 0:1].reshape([-1, m]) * pmd1_in
    pmdy_in = pm.hidden.weight.data[:, 1:2].reshape([-1, m]) * pmd1_in

    c1md1_b = c1m(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2md1_b = c2m(xb)[2]
    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    phimd1_b = phim(xb)[2]
    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    u1m_b = um(xb)[1]
    u2m_b = um(xb)[2]

    c1m_i = c1m(xi)[0]
    c2m_i = c2m(xi)[0]
    u1m_i = um(xi)[1]
    u2m_i = um(xi)[2]

    diffc1, diffc2, diffphi, diffu1, diffu2, diffp = 1, 1, 1, 1, 1, 1
    ttt = 0

    while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold or diffu1 > threshold or diffu2 > threshold or diffp > threshold) and ttt<ite:
        c1_before, c2_before, phi_before, u1_before, u2_before, p_before= c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]

        # NP
        u1m_in_1 = um(xy_inter)[3]
        u2m_in_1 = um(xy_inter)[4]
        phimdx_in_1 = phimdx_in@phim.predict.weight.data.T
        phimdy_in_1 = phimdy_in@phim.predict.weight.data.T
        phimdxx_in_1 = phimdxx_in@phim.predict.weight.data.T
        phimdyy_in_1 = phimdyy_in@phim.predict.weight.data.T

        A1 = (c1mdt_in + (u1m_in_1*c1mdx_in + u2m_in_1*c1mdy_in) - D1*(c1mdxx_in + c1mdyy_in + z1*(c1m_in*(phimdxx_in_1 + phimdyy_in_1) + c1mdx_in*phimdx_in_1 + c1mdy_in*phimdy_in_1))).detach().cpu().numpy()
        B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
        b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt_in + (u1m_in_1*c2mdx_in + u2m_in_1*c2mdy_in) - D2*(c2mdxx_in + c2mdyy_in + z2*(c2m_in*(phimdxx_in_1 + phimdyy_in_1) + c2mdx_in*phimdx_in_1 + c2mdy_in*phimdy_in_1))).detach().cpu().numpy()
        B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
        b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        # P
        c1m_in_1 = c1m(xy_inter)[1]
        c2m_in_1 = c2m(xy_inter)[1]

        A3 = -eps*(phimdxx_in + phimdyy_in).detach().cpu().numpy()
        fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + (z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1*B3, E1)) 
        b_P = np.vstack((fa3, p1*fb3, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        # NS
        phimdx_1 = phimdx_in@phim.predict.weight.data.T
        phimdy_1 = phimdy_in@phim.predict.weight.data.T

        C1[:, 0:m] = (u1mdt_in + u1m_in_1*u1mdx_in + u2m_in_1*u1mdy_in - nv*(u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
        C1[:, m:2*m] = (pmdx_in).detach().cpu().numpy()
        fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
        C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
        fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
        B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
        I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
        I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

        A_NS = np.vstack((C1, C2, p1*B4, p1*B5, p1*I3, p1*I4, E2)) 
        b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, p1*fi3, p1*fi4, fe2))
        # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
        wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
        um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
        pm.predict.weight.data = torch.from_numpy(wu_NS[m:2*m]).T.to(device)

        c1_after, c2_after, phi_after, u1_after, u2_after, p_after = c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]
        diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
        diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
        diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
        diffu1 = err_calculate.err_L2_3d(u1_after, u1_before, wr, a, b, c, d, t1, t2)
        diffu2 = err_calculate.err_L2_3d(u2_after, u2_before, wr, a, b, c, d, t1, t2)
        diffp = err_calculate.err_L2_3d(p_after, p_before, wr, a, b, c, d, t1, t2)

        ttt = ttt + 1
    print(ttt)

    end = time.perf_counter()
    total_time += (end - start)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        x2_re.requires_grad_(True)
        c1m_plot_1 = c1m(x2_re)[1]
        c2m_plot_1 = c2m(x2_re)[1]
        phim_plot_1 = phim(x2_re)[1]
        phimd1_plot = phim(x2_re)[2]
        u1m_plot_1 = um(x2_re)[3]
        u2m_plot_1 = um(x2_re)[4]
        pm_plot_1 = pm(x2_re)[1]
        results['c1_error'][i] = err_calculate.err_L2_2d(c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
        results['c2_error'][i] = err_calculate.err_L2_2d(c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
        results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
        results['u1_error'][i] = err_calculate.err_L2_2d(u1m_plot_1, ex1_real.u1(x2_re), wr2, a, b, c, d)
        results['u2_error'][i] = err_calculate.err_L2_2d(u2m_plot_1, ex1_real.u2(x2_re), wr2, a, b, c, d)
        results['p_error'][i] = err_calculate.err_L2_2d(pm_plot_1, ex1_real.p(x2_re), wr2, a, b, c, d)
        results['c1_min'][i] = torch.min(c1m_plot_1)
        results['c2_min'][i] = torch.min(c2m_plot_1)
        results['m1'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
        results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot_1, u2m_plot_1, x2_re, wr2, a, b, c, d)
        phidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        phidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        energy_terms = c1m_plot_1 * torch.log(c1m_plot_1) + c2m_plot_1 * torch.log(c2m_plot_1) + eps ** 2 / 2 * (phidx ** 2 + phidy ** 2) + 0.5*(u1m_plot_1**2 + u2m_plot_1**2)
        real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps**2/2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2) + 0.5*(ex1_real.u1(x2_re)**2 + ex1_real.u2(x2_re)**2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
        results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

    print(f'c1_L^2 error: {results["c1_error"][-1]}')
    print(f'c2_L^2 error: {results["c2_error"][-1]}')
    print(f'phi_L^2 error: {results["phi_error"][-1]}')
    print(f'u1_L^2 error: {results["u1_error"][-1]}')
    print(f'u2_L^2 error: {results["u2_error"][-1]}')
    print(f'p_L^2 error: {results["p_error"][-1]}')

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['c1_error_all'][0:Nre] = results['c1_error']
    results_all['c2_error_all'][0:Nre] = results['c2_error']
    results_all['phi_error_all'][0:Nre] = results['phi_error']
    results_all['u1_error_all'][0:Nre] = results['u1_error']
    results_all['u2_error_all'][0:Nre] = results['u2_error']
    results_all['p_error_all'][0:Nre] = results['p_error']
    results_all['Divergence_free_all'][0:Nre] = results['Divergence_free']

    PNPNS_plot_st(xt_re, results, 0)

    t1 = t2

    for k in range(Nt-1):

        start = time.perf_counter()

        t2 = t1 + s
        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        N_in = xy_inter.shape[0]
        N_b = xb.shape[0]
        N_i = xi.shape[0]
        C1 = np.zeros((N_in, 2 * m)) 
        C2 = np.zeros((N_in, 2 * m))
        B4 = np.zeros((N_b, 2 * m))   # u1
        B5 = np.zeros((N_b, 2 * m))   # u2
        I3 = np.zeros((N_i, 2 * m))   # u1
        I4 = np.zeros((N_i, 2 * m))   # u2

        fa1 = (ex1_real.f1(xy_inter, z1, D1)).detach().cpu().numpy()
        fa2 = (ex1_real.f2(xy_inter, z2, D2)).detach().cpu().numpy()
        fb1 = (ex1_real.c1x(xb) * n1 + ex1_real.c1y(xb) * n2).detach().cpu().numpy()
        fb2 = (ex1_real.c2x(xb) * n1 + ex1_real.c2y(xb) * n2).detach().cpu().numpy()
        fb3 = (ex1_real.phix(xb) * n1 + ex1_real.phiy(xb) * n2).detach().cpu().numpy()
        fb4 = (ex1_real.u1(xb)).detach().cpu().numpy()
        fb5 = (ex1_real.u2(xb)).detach().cpu().numpy()
        fi1 = (c1m(xi)[1]).detach().cpu().numpy()
        fi2 = (c2m(xi)[1]).detach().cpu().numpy()
        fi3 = (um(xi)[3]).detach().cpu().numpy()
        fi4 = (um(xi)[4]).detach().cpu().numpy()

        RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(um, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(pm, m, a, b, c, d, t1, t2)

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        E2 = np.zeros((n, 2 * m))
        fe2 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_p = torch.zeros(n_int ** 2, 2 * m)
                output_phi = phim(x2_t)[0]
                exact_phi = ex1_real.phi(x2_t)
                output_p[:, m:2 * m] = pm(x2_t)[0]
                exact_p = ex1_real.p(x2_t)
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
                fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()
                E2[j:j + 1, :] = (scale_factor * torch.sum(output_p * wr2, dim=0,keepdim=True)).detach().cpu().numpy()
                fe2[j:j + 1, :] = (scale_factor * torch.sum(exact_p * wr2, dim=0)).detach().cpu().numpy()

        c1m_in, c1md1_in, c1md2_in, c1mdx_in, c1mdy_in, c1mdt_in, c1mdxx_in, c1mdyy_in = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
        c2m_in, c2md1_in, c2md2_in, c2mdx_in, c2mdy_in, c2mdt_in, c2mdxx_in, c2mdyy_in = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
        phim_in, phimd1_in, phimd2_in, phimdx_in, phimdy_in, phimdt_in, phimdxx_in, phimdyy_in = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

        umd1_in = um(xy_inter)[5]
        umd2_in = um(xy_inter)[6]
        u1mdx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u1mdy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd1_in
        u1mdt_in = um.hidden.weight.data[:, 2:3].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u2mdx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * umd1_in
        u2mdy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd1_in
        u2mdt_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 2:3].reshape([-1, m]) * umd1_in
        u1mdxx_in = um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * umd2_in
        u1mdyy_in = um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 3 * umd2_in
        u2mdxx_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 3 * umd2_in
        u2mdyy_in = -um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * umd2_in

        pmd1_in = pm(xy_inter)[2]
        pmdx_in = pm.hidden.weight.data[:, 0:1].reshape([-1, m]) * pmd1_in
        pmdy_in = pm.hidden.weight.data[:, 1:2].reshape([-1, m]) * pmd1_in

        c1md1_b = c1m(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2md1_b = c2m(xb)[2]
        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        phimd1_b = phim(xb)[2]
        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        u1m_b = um(xb)[1]
        u2m_b = um(xb)[2]

        c1m_i = c1m(xi)[0]
        c2m_i = c2m(xi)[0]
        u1m_i = um(xi)[1]
        u2m_i = um(xi)[2]

        diffc1, diffc2, diffphi, diffu1, diffu2, diffp = 1, 1, 1, 1, 1, 1
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold or diffu1 > threshold or diffu2 > threshold or diffp > threshold) and ttt<ite:
            c1_before, c2_before, phi_before, u1_before, u2_before, p_before= c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]

            # NP
            u1m_in_1 = um(xy_inter)[3]
            u2m_in_1 = um(xy_inter)[4]
            phimdx_in_1 = phimdx_in@phim.predict.weight.data.T
            phimdy_in_1 = phimdy_in@phim.predict.weight.data.T
            phimdxx_in_1 = phimdxx_in@phim.predict.weight.data.T
            phimdyy_in_1 = phimdyy_in@phim.predict.weight.data.T

            A1 = (c1mdt_in + (u1m_in_1*c1mdx_in + u2m_in_1*c1mdy_in) - D1*(c1mdxx_in + c1mdyy_in + z1*(c1m_in*(phimdxx_in_1 + phimdyy_in_1) + c1mdx_in*phimdx_in_1 + c1mdy_in*phimdy_in_1))).detach().cpu().numpy()
            B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
            b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt_in + (u1m_in_1*c2mdx_in + u2m_in_1*c2mdy_in) - D2*(c2mdxx_in + c2mdyy_in + z2*(c2m_in*(phimdxx_in_1 + phimdyy_in_1) + c2mdx_in*phimdx_in_1 + c2mdy_in*phimdy_in_1))).detach().cpu().numpy()
            B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
            b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            # P
            c1m_in_1 = c1m(xy_inter)[1]
            c2m_in_1 = c2m(xy_inter)[1]

            A3 = -eps*(phimdxx_in + phimdyy_in).detach().cpu().numpy()
            fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + (z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

            A_P = np.vstack((A3, p1*B3, E1)) 
            b_P = np.vstack((fa3, p1*fb3, fe1))
            wu_P = np.linalg.lstsq(A_P, b_P)[0] 
            phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

            # NS
            phimdx_1 = phimdx_in@phim.predict.weight.data.T
            phimdy_1 = phimdy_in@phim.predict.weight.data.T

            C1[:, 0:m] = (u1mdt_in + u1m_in_1*u1mdx_in + u2m_in_1*u1mdy_in - nv*(u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
            C1[:, m:2*m] = (pmdx_in).detach().cpu().numpy()
            fc1 = (ex1_real.g1(xy_inter, z1, z2, nv) - phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
            C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
            fc2 = (ex1_real.g2(xy_inter, z1, z2, nv) - phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
            B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
            I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
            I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

            A_NS = np.vstack((C1, C2, p1*B4, p1*B5, p1*I3, p1*I4, E2)) 
            b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, p1*fi3, p1*fi4, fe2))
            # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
            wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
            um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
            pm.predict.weight.data = torch.from_numpy(wu_NS[m:2*m]).T.to(device)

            c1_after, c2_after, phi_after, u1_after, u2_after, p_after = c1m(x)[1], c2m(x)[1], phim(x)[1], um(x)[3], um(x)[4], pm(x)[1]
            diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
            diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
            diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
            diffu1 = err_calculate.err_L2_3d(u1_after, u1_before, wr, a, b, c, d, t1, t2)
            diffu2 = err_calculate.err_L2_3d(u2_after, u2_before, wr, a, b, c, d, t1, t2)
            diffp = err_calculate.err_L2_3d(p_after, p_before, wr, a, b, c, d, t1, t2)

            ttt = ttt + 1

        print(ttt)

        end = time.perf_counter()
        total_time += (end - start)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            x2_re.requires_grad_(True)
            c1m_plot_1 = c1m(x2_re)[1]
            c2m_plot_1 = c2m(x2_re)[1]
            phim_plot_1 = phim(x2_re)[1]
            phimd1_plot = phim(x2_re)[2]
            u1m_plot_1 = um(x2_re)[3]
            u2m_plot_1 = um(x2_re)[4]
            pm_plot_1 = pm(x2_re)[1]
            results['c1_error'][i] = err_calculate.err_L2_2d(c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
            results['c2_error'][i] = err_calculate.err_L2_2d(c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
            results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
            results['u1_error'][i] = err_calculate.err_L2_2d(u1m_plot_1, ex1_real.u1(x2_re), wr2, a, b, c, d)
            results['u2_error'][i] = err_calculate.err_L2_2d(u2m_plot_1, ex1_real.u2(x2_re), wr2, a, b, c, d)
            results['p_error'][i] = err_calculate.err_L2_2d(pm_plot_1, ex1_real.p(x2_re), wr2, a, b, c, d)
            results['c1_min'][i] = torch.min(c1m_plot_1)
            results['c2_min'][i] = torch.min(c2m_plot_1)
            results['m1'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
            results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot_1, u2m_plot_1, x2_re, wr2, a, b, c, d)
            phidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            phidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            energy_terms = c1m_plot_1 * torch.log(c1m_plot_1) + c2m_plot_1 * torch.log(c2m_plot_1) + eps ** 2 / 2 * (phidx ** 2 + phidy ** 2) + 0.5*(u1m_plot_1**2 + u2m_plot_1**2)
            real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps**2/2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2) + 0.5*(ex1_real.u1(x2_re)**2 + ex1_real.u2(x2_re)**2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
            results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

        print(f'c1_L^2 error: {results["c1_error"][-1]}')
        print(f'c2_L^2 error: {results["c2_error"][-1]}')
        print(f'phi_L^2 error: {results["phi_error"][-1]}')
        print(f'u1_L^2 error: {results["u1_error"][-1]}')
        print(f'u2_L^2 error: {results["u2_error"][-1]}')
        print(f'p_L^2 error: {results["p_error"][-1]}')

        PNPNS_plot_st(xt_re, results, k+1)

        xt_re_all[(k + 1) * Nre:(k + 2) * Nre] = xt_re
        results_all['e_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e']
        results_all['m1_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1']
        results_all['m2_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2']
        results_all['c1_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_min']
        results_all['c2_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_min']
        results_all['c1_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_error']
        results_all['c2_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_error']
        results_all['phi_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['phi_error']
        results_all['u1_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['u1_error']
        results_all['u2_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['u2_error']
        results_all['p_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['p_error']
        results_all['Divergence_free_all'][(k + 1) * Nre:(k + 2) * Nre] = results['Divergence_free']

        t1 = t2

    PNPNS_plot_st_all(xt_re_all, results_all)

    print(total_time)
