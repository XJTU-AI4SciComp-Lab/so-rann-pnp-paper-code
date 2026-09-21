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
from RNN_ref import RNN_funs
import torch
from RNN_ref import sampling_points
import time

torch.set_default_dtype(torch.float64)


def _refit_phi_after_mass_correction(
    phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
    A_P, fb3, fe1, p1, eps, z1, z2, m, device,
):
    """Fit the reported potential to the final mass-corrected concentrations."""
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
    return weights, scale1_in, scale2_in

def PNP_plot_st_re(xt_re, results, k):
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

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flat

    plots_config = [
        (ax1, 'c1_error', r'$c_1$ error'),
        (ax2, 'c2_error', r'$c_2$ error'),
        (ax3, 'phi_error', r'$\phi$ error'),
        (ax4, 'c1_min', r'$c_1$ min'),
        (ax5, 'c2_min', r'$c_2$ min'),
        (ax6, 'm1', r'$m_1$'),
        (ax7, 'm2', r'$m_2$')
    ]

    for ax, data_key, title in plots_config:
        if data_key in ('m1', 'm2', 'm1_all', 'm2_all'):
            ax.plot(xt_re, results[data_key], label='numerical')
            real_key = data_key.replace('_all', '_real_all') if data_key.endswith('_all') else data_key + '_real'
            if real_key in results:
                ax.plot(xt_re, results[real_key], linestyle='--', label='real')
            ax.legend(
                loc='upper right',
                fontsize=8,
                frameon=True,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=1.4,
                labelspacing=0.25
            )
        else:
            ax.plot(xt_re, results[data_key])

        ax.set_xlabel('$t$')

        if 'error' in data_key:
            ax.set_ylabel('$L_2$ error')
        elif 'min' in data_key:
            ax.set_ylabel('minimum')
        else:  # m1, m2
            ax.set_ylabel('mass')

        ax.set_title(title)
        ax.grid(True, color='0.85', linewidth=0.5)

    ax8.plot(xt_re, results['e'], label='numerical')
    ax8.plot(xt_re, results['e_real'], linestyle='--', label='real')
    ax8.set(xlabel='$t$', ylabel='$E$', title='energy')
    ax8.set_ylim(top=2.5) 
    ax8.legend(
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.4,
        labelspacing=0.25
    )
    ax8.grid(True, color='0.85', linewidth=0.5)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex1_paper/PNP_r_ex1_paper/SO-RaNN/PNP-st-re-ex1_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_all(xt_re, results):
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


    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flat

    plots_config = [
        (ax1, 'c1_error_all', r'$c_1$ error'),
        (ax2, 'c2_error_all', r'$c_2$ error'),
        (ax3, 'phi_error_all', r'$\phi$ error'),
        (ax4, 'c1_min_all', r'$c_1$ min'),
        (ax5, 'c2_min_all', r'$c_2$ min'),
        (ax6, 'm1_all', r'$m_1$'),
        (ax7, 'm2_all', r'$m_2$'),
    ]

    for ax, data_key, title in plots_config:
        if data_key in ('m1', 'm2', 'm1_all', 'm2_all'):
            ax.plot(xt_re, results[data_key], label='numerical')
            real_key = data_key.replace('_all', '_real_all') if data_key.endswith('_all') else data_key + '_real'
            if real_key in results:
                ax.plot(xt_re, results[real_key], linestyle='--', label='real')
            ax.legend(
                loc='upper right',
                fontsize=8,
                frameon=True,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=1.4,
                labelspacing=0.25
            )
        else:
            ax.plot(xt_re, results[data_key])

        ax.set_xlabel('$t$')

        if 'error' in data_key:
            ax.set_ylabel('$L_2$ error')
        elif 'min' in data_key:
            ax.set_ylabel('minimum')
        else:  # m1, m2
            ax.set_ylabel('mass')

        ax.set_title(title)
        ax.grid(True, color='0.85', linewidth=0.5)

    ax8.plot(xt_re, results['e_all'], label='numerical')
    ax8.plot(xt_re, results['e_real_all'], linestyle='--', label='real')
    ax8.set(xlabel='$t$', ylabel='$E$', title='energy')
    ax8.set_ylim(top=2.5) 
    ax8.legend(
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.4,
        labelspacing=0.25
    )
    ax8.grid(True, color='0.85', linewidth=0.5)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex1_paper/PNP_r_ex1_paper/SO-RaNN/PNP-st-re-ex1-all.eps'), bbox_inches='tight')
    plt.close(fig)


def PNP_plot_st(xt_re, results, k):
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

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flat

    plots_config = [
        (ax1, 'c1_error', r'$c_1$ error'),
        (ax2, 'c2_error', r'$c_2$ error'),
        (ax3, 'phi_error', r'$\phi$ error'),
        (ax4, 'c1_min', r'$c_1$ min'),
        (ax5, 'c2_min', r'$c_2$ min'),
        (ax6, 'm1', r'$m_1$'),
        (ax7, 'm2', r'$m_2$')
    ]

    for ax, data_key, title in plots_config:
        if data_key in ('m1', 'm2', 'm1_all', 'm2_all'):
            ax.plot(xt_re, results[data_key], label='numerical')
            real_key = data_key.replace('_all', '_real_all') if data_key.endswith('_all') else data_key + '_real'
            if real_key in results:
                ax.plot(xt_re, results[real_key], linestyle='--', label='real')
            ax.legend(
                loc='upper right',
                fontsize=8,
                frameon=True,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=1.4,
                labelspacing=0.25
            )
        else:
            ax.plot(xt_re, results[data_key])

        ax.set_xlabel('$t$')

        if 'error' in data_key:
            ax.set_ylabel('$L_2$ error')
        elif 'min' in data_key:
            ax.set_ylabel('minimum')
        else:  # m1, m2
            ax.set_ylabel('mass')

        ax.set_title(title)
        ax.grid(True, color='0.85', linewidth=0.5)

    ax8.plot(xt_re, results['e'], label='numerical')
    ax8.plot(xt_re, results['e_real'], linestyle='--', label='real')
    ax8.set(xlabel='$t$', ylabel='$E$', title='energy')
    ax8.set_ylim(top=2.5) 
    ax8.legend(
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.4,
        labelspacing=0.25
    )
    ax8.grid(True, color='0.85', linewidth=0.5)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex1_paper/PNP_r_ex1_paper/RaNN/PNP-st-ex1_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_all(xt_re, results):
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


    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flat

    plots_config = [
        (ax1, 'c1_error_all', r'$c_1$ error'),
        (ax2, 'c2_error_all', r'$c_2$ error'),
        (ax3, 'phi_error_all', r'$\phi$ error'),
        (ax4, 'c1_min_all', r'$c_1$ min'),
        (ax5, 'c2_min_all', r'$c_2$ min'),
        (ax6, 'm1_all', r'$m_1$'),
        (ax7, 'm2_all', r'$m_2$'),
    ]

    for ax, data_key, title in plots_config:
        if data_key in ('m1', 'm2', 'm1_all', 'm2_all'):
            ax.plot(xt_re, results[data_key], label='numerical')
            real_key = data_key.replace('_all', '_real_all') if data_key.endswith('_all') else data_key + '_real'
            if real_key in results:
                ax.plot(xt_re, results[real_key], linestyle='--', label='real')
            ax.legend(
                loc='upper right',
                fontsize=8,
                frameon=True,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=1.4,
                labelspacing=0.25
            )
        else:
            ax.plot(xt_re, results[data_key])

        ax.set_xlabel('$t$')

        if 'error' in data_key:
            ax.set_ylabel('$L_2$ error')
        elif 'min' in data_key:
            ax.set_ylabel('minimum')
        else:  # m1, m2
            ax.set_ylabel('mass')

        ax.set_title(title)
        ax.grid(True, color='0.85', linewidth=0.5)

    ax8.plot(xt_re, results['e_all'], label='numerical')
    ax8.plot(xt_re, results['e_real_all'], linestyle='--', label='real')
    ax8.set(xlabel='$t$', ylabel='$E$', title='energy')
    ax8.set_ylim(top=2.5) 
    ax8.legend(
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.4,
        labelspacing=0.25
    )
    ax8.grid(True, color='0.85', linewidth=0.5)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex1_paper/PNP_r_ex1_paper/RaNN/PNP-st-ex1-all.eps'), bbox_inches='tight')
    plt.close(fig)

def pnp_ex1_picard_st_decoupled_remass(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, ite, p1):
    device = 'cpu'
    total_time1 = 0        
    total_time2 = 0        
    total_time3 = 0        
    picard_iterations = []

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre),
        'm1_raw': np.zeros(Nre), 'm2_raw': np.zeros(Nre),
        'm1_cut': np.zeros(Nre), 'm2_cut': np.zeros(Nre),
        'm1_real': np.zeros(Nre), 'm2_real': np.zeros(Nre),
        'e': np.zeros(Nre), 'e_real': np.zeros(Nre),
        'c1_error': np.zeros(Nre), 'c2_error': np.zeros(Nre),
        'phi_error': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre)
    }
    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt),
        'm1_raw_all': np.zeros(Nre * Nt), 'm2_raw_all': np.zeros(Nre * Nt),
        'm1_cut_all': np.zeros(Nre * Nt), 'm2_cut_all': np.zeros(Nre * Nt),
        'm1_real_all': np.zeros(Nre * Nt), 'm2_real_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'e_real_all': np.zeros(Nre * Nt),
        'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt),
        'c1_error_all': np.zeros(Nre * Nt), 'c2_error_all': np.zeros(Nre * Nt), 'phi_error_all': np.zeros(Nre * Nt)
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

    fa1 = ex1_real.f1(xy_inter, D1, z1).cpu().numpy()
    fa2 = ex1_real.f2(xy_inter, D2, z2).cpu().numpy()
    # fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).cpu().numpy()          # Neumann
    fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2 + z1*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c1(xb)).cpu().numpy()         # Robin
    # fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).cpu().numpy()          # Neumann
    fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2 + z2*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c2(xb)).cpu().numpy()         # Robin
    fb3 = (ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2).cpu().numpy()
    fi1 = ex1_real.c1(xi).cpu().numpy()
    fi2 = ex1_real.c2(xi).cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_phi = phim(x2_t)[0]
            exact_phi = ex1_real.phi(x2_t)
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
            fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()

    c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
    c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
    phim_in, phimd1, phimd2, phimdx, phimdy, _, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

    c1m_b, c2m_b, c1md1_b, c2md1_b, phimd1_b = c1m(xb)[0], c2m(xb)[0], c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    c1m_i, c2m_i = c1m(xi)[0], c2m(xi)[0]
    
    end1 = time.perf_counter()
    total_time1 += (end1 - start1)

    diffc1, diffc2, diffphi = 1, 1, 1
    diff_list = {'c1': [], 'c2': [], 'phi': []}
    ttt = 0
    
    while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
        start2 = time.perf_counter()
        c1_before, c2_before, phi_before = c1m(x)[1], c2m(x)[1], phim(x)[1]
        phimdx_b_before = phimdx_b@phim.predict.weight.data.T
        phimdy_b_before = phimdy_b@phim.predict.weight.data.T

        # NP
        phimdx_1 = phimdx@phim.predict.weight.data.T
        phimdy_1 = phimdy@phim.predict.weight.data.T
        phimdxx_1 = phimdxx@phim.predict.weight.data.T
        phimdyy_1 = phimdyy@phim.predict.weight.data.T

        A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        # B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()    # Neumann
        B1 = (c1mdx_b*n1 + c1mdy_b*n2 + z1*(phimdx_b_before*n1 + phimdy_b_before*n2)*c1m_b).detach().cpu().numpy()    # Robin
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
        b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        # B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()    # Neumann
        B2 = (c2mdx_b*n1 + c2mdy_b*n2 + z2*(phimdx_b_before*n1 + phimdy_b_before*n2)*c2m_b).detach().cpu().numpy()    # Robin
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
        b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        # P
        c1m_1_in, c2m_1_in = c1m(xy_inter)[1], c2m(xy_inter)[1]

        A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
        fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
        B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1*B3, E1)) 
        b_P = np.vstack((fa3, p1*fb3, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        c1_after, c2_after, phi_after = c1m(x)[1], c2m(x)[1], phim(x)[1]
        diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
        diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
        diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
        diff_list['c1'].append(float(diffc1.item()))
        diff_list['c2'].append(float(diffc2.item()))
        diff_list['phi'].append(float(diffphi.item()))

        ttt = ttt + 1

        end2 = time.perf_counter()
        total_time2 += (end2 - start2)
    print(ttt)
    picard_iterations.append(int(ttt))

    
    start3 = time.perf_counter()

    mass1_re = torch.ones(Nre)
    mass2_re = torch.ones(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re

    # The final reported potential must be consistent with the concentrations
    # after both positivity truncation and mass correction.
    reported_phi_weights, scale1_in, scale2_in = _refit_phi_after_mass_correction(
        phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
        A_P, fb3, fe1, p1, eps, z1, z2, m, device,
    )
    
    end3 = time.perf_counter()
    total_time3 += (end3 - start3)

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
    # c1m_1_in = s_c1_in_0 * c1m(xy_inter)[1]
    # c2m_1_in = s_c2_in_0 * c2m(xy_inter)[1]
    #
    # A3 = -eps * (phimdxx + phimdyy).detach().cpu().numpy()
    # fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1 * c1m_1_in + z2 * c2m_1_in).detach().cpu().numpy()
    # B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()
    #
    # b_P = np.vstack((fa3, p1 * fb3, fe1))
    # phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)



    # s_c1_int_0 = torch.from_numpy(s_c1_linear_0(xt_re.squeeze().numpy()))
    # s_c2_int_0 = torch.from_numpy(s_c2_linear_0(xt_re.squeeze().numpy()))

    # s_c1_int_0 = s_c1_cubic_0(xt_re.squeeze().numpy())
    # s_c2_int_0 = s_c2_cubic_0(xt_re.squeeze().numpy())

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        c1_hidden, c1m_plot_1 = c1m(x2_re)[0], c1m(x2_re)[1]
        c2_hidden, c2m_plot_1 = c2m(x2_re)[0], c2m(x2_re)[1]
        c1_raw = c1m.predict(c1_hidden)
        c2_raw = c2m.predict(c2_hidden)
        phim_plot_1 = phim(x2_re)[1]
        phimd1_plot = phim(x2_re)[2]
        results['c1_error'][i] = err_calculate.err_L2_2d(s_c1[i]*c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
        results['c2_error'][i] = err_calculate.err_L2_2d(s_c2[i]*c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
        results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
        results['c1_min'][i] = torch.min(s_c1[i] * c1m_plot_1)
        results['c2_min'][i] = torch.min(s_c2[i] * c2m_plot_1)
        results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m_plot_1, wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m_plot_1, wr2, a, b, c, d)
        results['m1_raw'][i] = err_calculate.int_2d(c1_raw, wr2, a, b, c, d)
        results['m2_raw'][i] = err_calculate.int_2d(c2_raw, wr2, a, b, c, d)
        results['m1_cut'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
        results['m2_cut'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
        results['m1_real'][i] = err_calculate.int_2d(ex1_real.c1(x2_re), wr2, a, b, c, d).item()
        results['m2_real'][i] = err_calculate.int_2d(ex1_real.c2(x2_re), wr2, a, b, c, d).item()
        psidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        psidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
        energy_terms = s_c1[i]*c1m_plot_1 * torch.log(s_c1[i]*c1m_plot_1) + s_c2[i]*c2m_plot_1 * torch.log(s_c2[i]*c2m_plot_1) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
        real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps ** 2 / 2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
        results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

    print(f'c1_L^2 error: {results["c1_error"][-1]}')
    print(f'c2_L^2 error: {results["c2_error"][-1]}')
    print(f'phi_L^2 error: {results["phi_error"][-1]}')

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['e_real_all'][0:Nre] = results['e_real']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['m1_raw_all'][0:Nre] = results['m1_raw']
    results_all['m2_raw_all'][0:Nre] = results['m2_raw']
    results_all['m1_cut_all'][0:Nre] = results['m1_cut']
    results_all['m2_cut_all'][0:Nre] = results['m2_cut']
    results_all['m1_real_all'][0:Nre] = results['m1_real']
    results_all['m2_real_all'][0:Nre] = results['m2_real']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['c1_error_all'][0:Nre] = results['c1_error']
    results_all['c2_error_all'][0:Nre] = results['c2_error']
    results_all['phi_error_all'][0:Nre] = results['phi_error']

    PNP_plot_st_re(xt_re, results, 0)

    t1 = t2

    for k in range(Nt-1):

        start1 = time.perf_counter()

        t2 = t1 + s

        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        fa1 = ex1_real.f1(xy_inter, D1, z1).cpu().numpy()
        fa2 = ex1_real.f2(xy_inter, D2, z2).cpu().numpy()
        # fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).cpu().numpy()          # Neumann
        fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2 + z1*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c1(xb)).cpu().numpy()         # Robin
        # fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).cpu().numpy()          # Neumann
        fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2 + z2*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c2(xb)).cpu().numpy()         # Robin
        fb3 = (ex1_real.phix(xb) * n1 + ex1_real.phiy(xb) * n2).cpu().numpy()
        fi1 = (s_c1[-1] * c1m(xi)[1]).detach().cpu().numpy()
        fi2 = (s_c2[-1] * c2m(xi)[1]).detach().cpu().numpy()

        RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_phi = phim(x2_t)[0]
                exact_phi = ex1_real.phi(x2_t)
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
                fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()

        c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
        c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
        phim_in, phimd1, phimd2, phimdx, phimdy, phimdt, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

        c1m_b, c2m_b, c1md1_b, c2md1_b, phimd1_b = c1m(xb)[0], c2m(xb)[0], c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        c1m_i, c2m_i = c1m(xi)[0], c2m(xi)[0]

        end1 = time.perf_counter()
        total_time1 += (end1 - start1)

        diffc1, diffc2, diffphi = 1, 1, 1
        diff_list = {'c1': [], 'c2': [], 'phi': []}
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
            start2 = time.perf_counter()
            c1_before, c2_before, phi_before = c1m(x)[1], c2m(x)[1], phim(x)[1]
            phimdx_b_before = phimdx_b @ phim.predict.weight.data.T
            phimdy_b_before = phimdy_b @ phim.predict.weight.data.T

            # NP
            phimdx_1 = phimdx@phim.predict.weight.data.T
            phimdy_1 = phimdy@phim.predict.weight.data.T
            phimdxx_1 = phimdxx@phim.predict.weight.data.T
            phimdyy_1 = phimdyy@phim.predict.weight.data.T

            A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            # B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()    # Neumann
            B1 = (c1mdx_b * n1 + c1mdy_b * n2 + z1 * (phimdx_b_before * n1 + phimdy_b_before * n2) * c1m_b).detach().cpu().numpy()  # Robin
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
            b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            # B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()    # Neumann
            B2 = (c2mdx_b * n1 + c2mdy_b * n2 + z2 * (phimdx_b_before * n1 + phimdy_b_before * n2) * c2m_b).detach().cpu().numpy()  # Robin
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
            b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            # P
            c1m_1_in = c1m(xy_inter)[1]
            c2m_1_in = c2m(xy_inter)[1]

            A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
            fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
            B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

            A_P = np.vstack((A3, p1*B3, E1)) 
            b_P = np.vstack((fa3, p1*fb3, fe1))
            wu_P = np.linalg.lstsq(A_P, b_P)[0] 
            phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

            c1_after, c2_after, phi_after = c1m(x)[1], c2m(x)[1], phim(x)[1]
            diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
            diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
            diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)
            diff_list['c1'].append(float(diffc1.item()))
            diff_list['c2'].append(float(diffc2.item()))
            diff_list['phi'].append(float(diffphi.item()))

            ttt = ttt + 1
            end2 = time.perf_counter()
            total_time2 += (end2 - start2)
        print(ttt)
        picard_iterations.append(int(ttt))

        start3 = time.perf_counter()
        mass1_re = torch.ones(Nre)
        mass2_re = torch.ones(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re


        # Refit the potential after the final concentration correction so that
        # the state passed to diagnostics and subsequent blocks is consistent.
        reported_phi_weights, scale1_in, scale2_in = _refit_phi_after_mass_correction(
            phim, c1m, c2m, xy_inter, xt_re, s_c1, s_c2,
            A_P, fb3, fe1, p1, eps, z1, z2, m, device,
        )

        end3 = time.perf_counter()
        total_time3 += (end3 - start3)

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
        # c1m_1_in = s_c1_in_0 * c1m(xy_inter)[1]
        # c2m_1_in = s_c2_in_0 * c2m(xy_inter)[1]
        #
        # A3 = -eps * (phimdxx + phimdyy).detach().cpu().numpy()
        # fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1 * c1m_1_in + z2 * c2m_1_in).detach().cpu().numpy()
        # B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()
        #
        # b_P = np.vstack((fa3, p1 * fb3, fe1))
        # wphi = torch.transpose((torch.from_numpy(wu_P[0:m]).to(device)), 0, 1)
        # for name, para in phim.named_parameters():
        #     if 'predict.weight' in name:
        #         para.data = wphi

        # s_c1_int_0 = torch.from_numpy(s_c1_linear_0(xt_re.squeeze().numpy()))
        # s_c2_int_0 = torch.from_numpy(s_c2_linear_0(xt_re.squeeze().numpy()))

        # s_c1_int_0 = s_c1_cubic_0(xt_re.squeeze().numpy())
        # s_c2_int_0 = s_c2_cubic_0(xt_re.squeeze().numpy())

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            c1_hidden, c1m_plot_1 = c1m(x2_re)[0], c1m(x2_re)[1]
            c2_hidden, c2m_plot_1 = c2m(x2_re)[0], c2m(x2_re)[1]
            c1_raw = c1m.predict(c1_hidden)
            c2_raw = c2m.predict(c2_hidden)
            phim_plot_1 = phim(x2_re)[1]
            phimd1_plot = phim(x2_re)[2]
            results['c1_error'][i] = err_calculate.err_L2_2d(s_c1[i]*c1m_plot_1, ex1_real.c1(x2_re), wr2, a, b, c, d)
            results['c2_error'][i] = err_calculate.err_L2_2d(s_c2[i]*c2m_plot_1, ex1_real.c2(x2_re), wr2, a, b, c, d)
            results['phi_error'][i] = err_calculate.err_L2_2d(phim_plot_1, ex1_real.phi(x2_re), wr2, a, b, c, d)
            results['c1_min'][i] = torch.min(s_c1[i] * c1m_plot_1)
            results['c2_min'][i] = torch.min(s_c2[i] * c2m_plot_1)
            results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m_plot_1, wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m_plot_1, wr2, a, b, c, d)
            results['m1_raw'][i] = err_calculate.int_2d(c1_raw, wr2, a, b, c, d)
            results['m2_raw'][i] = err_calculate.int_2d(c2_raw, wr2, a, b, c, d)
            results['m1_cut'][i] = err_calculate.int_2d(c1m_plot_1, wr2, a, b, c, d)
            results['m2_cut'][i] = err_calculate.int_2d(c2m_plot_1, wr2, a, b, c, d)
            results['m1_real'][i] = err_calculate.int_2d(ex1_real.c1(x2_re), wr2, a, b, c, d).item()
            results['m2_real'][i] = err_calculate.int_2d(ex1_real.c2(x2_re), wr2, a, b, c, d).item()
            psidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            psidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_plot)@phim.predict.weight.data.T
            energy_terms = s_c1[i]*c1m_plot_1 * torch.log(s_c1[i]*c1m_plot_1) + s_c2[i]*c2m_plot_1 * torch.log(s_c2[i]*c2m_plot_1) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
            real_energy_terms = ex1_real.c1(x2_re)*torch.log(ex1_real.c1(x2_re)) + ex1_real.c2(x2_re)*torch.log(ex1_real.c2(x2_re)) + eps ** 2 / 2*(ex1_real.phix(x2_re)**2 + ex1_real.phiy(x2_re)**2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)
            results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d)

        print(f'c1_L^2 error: {results["c1_error"][-1]}')
        print(f'c2_L^2 error: {results["c2_error"][-1]}')
        print(f'phi_L^2 error: {results["phi_error"][-1]}')

        PNP_plot_st_re(xt_re, results, k+1)

        xt_re_all[(k + 1) * Nre:(k + 2) * Nre] = xt_re
        results_all['e_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e']
        results_all['e_real_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e_real']
        results_all['m1_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1']
        results_all['m2_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2']
        results_all['m1_raw_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1_raw']
        results_all['m2_raw_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2_raw']
        results_all['m1_cut_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1_cut']
        results_all['m2_cut_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2_cut']
        results_all['m1_real_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1_real']
        results_all['m2_real_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2_real']
        results_all['c1_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_min']
        results_all['c2_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_min']
        results_all['c1_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_error']
        results_all['c2_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_error']
        results_all['phi_error_all'][(k + 1) * Nre:(k + 2) * Nre] = results['phi_error']

        t1 = t2

    PNP_plot_st_re_all(xt_re_all, results_all)

    print(total_time1)
    print(total_time2)
    print(total_time3)
    total_time = total_time1 + total_time2 + total_time3
    print(total_time)

    # Diagnostic Poisson re-fits on the last temporal block.  They use the same
    # basis, boundary rows, and gauge rows, changing only the charge-density
    # field on the right-hand side.  The final stage is the reported potential;
    # only the additional raw/cut diagnostic fits are excluded from the timing.
    c1_hidden_in, c1_cut_in = c1m(xy_inter)[0], c1m(xy_inter)[1]
    c2_hidden_in, c2_cut_in = c2m(xy_inter)[0], c2m(xy_inter)[1]
    c1_raw_in = c1m.predict(c1_hidden_in)
    c2_raw_in = c2m.predict(c2_hidden_in)
    concentration_stages = {
        'raw': (c1_raw_in, c2_raw_in),
        'cut': (c1_cut_in, c2_cut_in),
        'final': (scale1_in * c1_cut_in, scale2_in * c2_cut_in),
    }
    phi_weights = {}
    g_in = ex1_real.f3(xy_inter, eps, z1, z2)
    for stage, (c1_stage, c2_stage) in concentration_stages.items():
        if stage == 'final':
            phi_weights[stage] = reported_phi_weights
            continue
        rhs_in = (g_in + z1 * c1_stage + z2 * c2_stage).detach().cpu().numpy()
        rhs = np.vstack((rhs_in, p1 * fb3, fe1))
        phi_weights[stage] = np.linalg.lstsq(A_P, rhs, rcond=None)[0][:m]

    x2_final = torch.cat((x2, xt_re[-1] * torch.ones(x2.shape[0], 1)), dim=1)
    phi_basis_final = phim(x2_final)[0].detach().cpu().numpy()
    phi_exact_final = ex1_real.phi(x2_final)
    phi_stage_values = {
        stage: torch.as_tensor(phi_basis_final @ weights, dtype=phi_exact_final.dtype)
        for stage, weights in phi_weights.items()
    }
    phi_errors = {
        stage: float(err_calculate.err_L2_2d(value, phi_exact_final, wr2, a, b, c, d))
        for stage, value in phi_stage_values.items()
    }
    phi_change = float(
        err_calculate.err_L2_2d(
            phi_stage_values['final'], phi_stage_values['cut'], wr2, a, b, c, d
        )
    )
    phi_final_norm = float(
        torch.sqrt(err_calculate.int_2d(phi_stage_values['final'] ** 2, wr2, a, b, c, d))
    )
    potential_diagnostics = {
        'last_block_final_time': float(xt_re[-1]),
        'phi_l2_error_raw_charge_fit': phi_errors['raw'],
        'phi_l2_error_cut_charge_fit': phi_errors['cut'],
        'phi_l2_error_final_charge_fit': phi_errors['final'],
        'relative_cut_to_final_phi_change': phi_change / max(phi_final_norm, np.finfo(float).eps),
    }

    return {
        'time': xt_re_all.detach().cpu().numpy(),
        'results': {key: np.asarray(value).copy() for key, value in results_all.items()},
        'picard_iterations': picard_iterations,
        'potential_diagnostics': potential_diagnostics,
        'timings': {
            'setup': float(total_time1),
            'picard': float(total_time2),
            'correction': float(total_time3),
            'total': float(total_time),
        },
    }
def pnp_ex1_picard_st_decoupled_t(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, ite, p1):
    device = 'cpu'
    total_time1 = 0
    total_time2 = 0
    picard_iterations = []

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre),
        'm1_real': np.zeros(Nre), 'm2_real': np.zeros(Nre),
        'e': np.zeros(Nre), 'e_real': np.zeros(Nre),
        'c1_error': np.zeros(Nre), 'c2_error': np.zeros(Nre),
        'phi_error': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre)
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt),
        'm1_real_all': np.zeros(Nre * Nt), 'm2_real_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'e_real_all': np.zeros(Nre * Nt),
        'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt),
        'c1_error_all': np.zeros(Nre * Nt), 'c2_error_all': np.zeros(Nre * Nt), 'phi_error_all': np.zeros(Nre * Nt)
    }

    start1 = time.perf_counter() 

    xr = sampling_points.get_GL_points_3d(n_int)
    wr = sampling_points.get_weights_3d(n_int).reshape([n_int ** 3, 1])
    x = torch.zeros_like(xr)
    x[:, 0:1] = (b - a) / 2 * xr[:, 0:1] + (a + b) / 2
    x[:, 1:2] = (d - c) / 2 * xr[:, 1:2] + (c + d) / 2

    t2 = t1 + s

    RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
    RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)

    N1 = int((d-c)*(b-a)*s*n**3)
    N2x, N2y = int((b-a)*s*n**2), int((d-c)*s*n**2)
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

    with torch.no_grad():
        fa1 = ex1_real.f1(xy_inter, D1, z1).cpu().numpy()
        fa2 = ex1_real.f2(xy_inter, D2, z2).cpu().numpy()
        # fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).cpu().numpy()         # Neumann
        fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2 + z1*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c1(xb)).cpu().numpy()         # Robin
        # fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).cpu().numpy()         # Neumann
        fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2 + z2*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c2(xb)).cpu().numpy()         # Robin
        fb3 = (ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2).cpu().numpy()
        fi1 = ex1_real.c1(xi).cpu().numpy()
        fi2 = ex1_real.c2(xi).cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_phi = phim(x2_t)[0]
            exact_phi = ex1_real.phi(x2_t)
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
            fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()

    c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
    c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
    phim_in, phimd1, phimd2, phimdx, phimdy, phimdt, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

    c1m_b, c2m_b, c1md1_b, c2md1_b, phimd1_b = c1m(xb)[0], c2m(xb)[0], c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    c1m_i, c2m_i = c1m(xi)[0], c2m(xi)[0]

    end1 = time.perf_counter()
    total_time1 += end1 - start1

    diffc1, diffc2, diffphi = 1, 1, 1
    diff_list = {'c1': [], 'c2': [], 'phi': []}
    ttt = 0

    while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
        start2 = time.perf_counter() 

        c1_before = c1m(x)[1]
        c2_before = c2m(x)[1]
        phi_before = phim(x)[1]
        phimdx_b_before = phimdx_b@phim.predict.weight.data.T
        phimdy_b_before = phimdy_b@phim.predict.weight.data.T

        # NP
        phimdx_1 = phimdx@phim.predict.weight.data.T
        phimdy_1 = phimdy@phim.predict.weight.data.T
        phimdxx_1 = phimdxx@phim.predict.weight.data.T
        phimdyy_1 = phimdyy@phim.predict.weight.data.T

        A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        # B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()    # Neumann
        B1 = (c1mdx_b*n1 + c1mdy_b*n2 + z1*(phimdx_b_before*n1 + phimdy_b_before*n2)*c1m_b).detach().cpu().numpy()    # Robin
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
        b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        # B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
        B2 = (c2mdx_b*n1 + c2mdy_b*n2 + z2*(phimdx_b_before*n1 + phimdy_b_before*n2)*c2m_b).detach().cpu().numpy()    # Robin
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
        b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        # P
        c1m_1_in, c2m_1_in = c1m(xy_inter)[1], c2m(xy_inter)[1]

        A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
        fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
        B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1*B3, E1)) 
        b_P = np.vstack((fa3, p1*fb3, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        c1_after, c2_after, phi_after = c1m(x)[1], c2m(x)[1], phim(x)[1]

        diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
        diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
        diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)

        diff_list['c1'].append(float(diffc1.item()))
        diff_list['c2'].append(float(diffc2.item()))
        diff_list['phi'].append(float(diffphi.item()))

        ttt = ttt + 1

        end2 = time.perf_counter()
        total_time2 += end2 - start2
    print(ttt)
    picard_iterations.append(int(ttt))

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)

        c1_pred, c2_pred, phi_pred = c1m(x2_re)[1], c2m(x2_re)[1], phim(x2_re)[1]
        c1_exact, c2_exact, phi_exact = ex1_real.c1(x2_re), ex1_real.c2(x2_re), ex1_real.phi(x2_re)

        results['c1_error'][i] = err_calculate.err_L2_2d(c1_pred, c1_exact, wr2, a, b, c, d).item()
        results['c2_error'][i] = err_calculate.err_L2_2d(c2_pred, c2_exact, wr2, a, b, c, d).item()
        results['phi_error'][i] = err_calculate.err_L2_2d(phi_pred, phi_exact, wr2, a, b, c, d).item()
        results['c1_min'][i] = torch.min(c1_pred).item()
        results['c2_min'][i] = torch.min(c2_pred).item()
        results['m1'][i] = err_calculate.int_2d(c1_pred, wr2, a, b, c, d).item()
        results['m2'][i] = err_calculate.int_2d(c2_pred, wr2, a, b, c, d).item()
        results['m1_real'][i] = err_calculate.int_2d(c1_exact, wr2, a, b, c, d).item()
        results['m2_real'][i] = err_calculate.int_2d(c2_exact, wr2, a, b, c, d).item()

        phim_output = phim(x2_re)
        psidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim_output[2]) @ phim.predict.weight.data.T
        psidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim_output[2]) @ phim.predict.weight.data.T

        energy_terms = c1_pred * torch.log(c1_pred) + c2_pred * torch.log(c2_pred) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
        real_energy_terms = c1_exact * torch.log(c1_exact) + c2_exact * torch.log(c2_exact) + eps ** 2 / 2 * (ex1_real.phix(x2_re) ** 2 + ex1_real.phiy(x2_re) ** 2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d).item()
        results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d).item()

    print(f'c1_L^2 error: {results["c1_error"][-1]}')
    print(f'c2_L^2 error: {results["c2_error"][-1]}')
    print(f'phi_L^2 error: {results["phi_error"][-1]}')

    PNP_plot_st(xt_re, results, 0)

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['e_real_all'][0:Nre] = results['e_real']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['m1_real_all'][0:Nre] = results['m1_real']
    results_all['m2_real_all'][0:Nre] = results['m2_real']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['c1_error_all'][0:Nre] = results['c1_error']
    results_all['c2_error_all'][0:Nre] = results['c2_error']
    results_all['phi_error_all'][0:Nre] = results['phi_error']

    t1 = t2

    for k in range(Nt-1):
        results = {
            'm1': np.zeros(Nre), 'm2': np.zeros(Nre),
            'm1_real': np.zeros(Nre), 'm2_real': np.zeros(Nre),
            'e': np.zeros(Nre), 'e_real': np.zeros(Nre),
            'c1_error': np.zeros(Nre), 'c2_error': np.zeros(Nre),
            'phi_error': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre)
        }

        start1 = time.perf_counter()

        t2 = t1 + s

        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        fa1 = ex1_real.f1(xy_inter, D1, z1).cpu().numpy()
        fa2 = ex1_real.f2(xy_inter, D2, z2).cpu().numpy()
        # fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2).cpu().numpy()         # Neumann
        fb1 = (ex1_real.c1x(xb)*n1 + ex1_real.c1y(xb)*n2 + z1*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c1(xb)).cpu().numpy()         # Robin
        # fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2).cpu().numpy()         # Neumann
        fb2 = (ex1_real.c2x(xb)*n1 + ex1_real.c2y(xb)*n2 + z2*(ex1_real.phix(xb)*n1 + ex1_real.phiy(xb)*n2)*ex1_real.c2(xb)).cpu().numpy()         # Robin
        fb3 = (ex1_real.phix(xb) * n1 + ex1_real.phiy(xb) * n2).cpu().numpy()
        # fi1 = ex1_real.c1(xi).cpu().numpy()
        # fi2 = ex1_real.c2(xi).cpu().numpy()
        fi1 = (c1m(xi)[1]).detach().cpu().numpy()
        fi2 = (c2m(xi)[1]).detach().cpu().numpy()

        RNN_funs.bias_re_3d(c1m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(c2m, m, a, b, c, d, t1, t2)
        RNN_funs.bias_re_3d(phim, m, a, b, c, d, t1, t2)

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_phi = phim(x2_t)[0]
                exact_phi = ex1_real.phi(x2_t)
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()
                fe1[j] = (scale_factor * torch.sum(exact_phi * wr2, dim=0)).cpu().numpy()

        c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
        c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
        phim_in, phimd1, phimd2, phimdx, phimdy, phimdt, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

        c1m_b, c2m_b, c1md1_b, c2md1_b, phimd1_b = c1m(xb)[0], c2m(xb)[0], c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        c1m_i = c1m(xi)[0]
        c2m_i = c2m(xi)[0]

        end1 = time.perf_counter()
        total_time1 += end1 - start1

        diffc1, diffc2, diffphi = 1, 1, 1
        diff_list = {'c1': [], 'c2': [], 'phi': []}
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
            start2 = time.perf_counter()

            c1_before, c2_before, phi_before = c1m(x)[1], c2m(x)[1], phim(x)[1]
            phimdx_b_before = phimdx_b @ phim.predict.weight.data.T
            phimdy_b_before = phimdy_b @ phim.predict.weight.data.T

            # NP
            phimdx_1 = phimdx@phim.predict.weight.data.T
            phimdy_1 = phimdy@phim.predict.weight.data.T
            phimdxx_1 = phimdxx@phim.predict.weight.data.T
            phimdyy_1 = phimdyy@phim.predict.weight.data.T

            A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            # B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()    # Neumann
            B1 = (c1mdx_b * n1 + c1mdy_b * n2 + z1 * (phimdx_b_before * n1 + phimdy_b_before * n2) * c1m_b).detach().cpu().numpy()  # Robin
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
            b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0]
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            # B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
            B2 = (c2mdx_b * n1 + c2mdy_b * n2 + z2 * (phimdx_b_before * n1 + phimdy_b_before * n2) * c2m_b).detach().cpu().numpy()  # Robin
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
            b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0]
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            # P
            c1m_1_in = c1m(xy_inter)[1]
            c2m_1_in = c2m(xy_inter)[1]

            A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
            fa3 = (ex1_real.f3(xy_inter, eps, z1, z2) + z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
            B3 = (phimdx_b*n1 + phimdy_b*n2).detach().cpu().numpy()

            A_P = np.vstack((A3, p1*B3, E1)) 
            b_P = np.vstack((fa3, p1*fb3, fe1))
            wu_P = np.linalg.lstsq(A_P, b_P)[0]
            phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

            c1_after, c2_after, phi_after = c1m(x)[1], c2m(x)[1], phim(x)[1]
            diffc1 = err_calculate.err_L2_3d(c1_after, c1_before, wr, a, b, c, d, t1, t2)
            diffc2 = err_calculate.err_L2_3d(c2_after, c2_before, wr, a, b, c, d, t1, t2)
            diffphi = err_calculate.err_L2_3d(phi_after, phi_before, wr, a, b, c, d, t1, t2)

            diff_list['c1'].append(float(diffc1.item()))
            diff_list['c2'].append(float(diffc2.item()))
            diff_list['phi'].append(float(diffphi.item()))

            ttt = ttt + 1

            end2 = time.perf_counter()
            total_time2 += end2 - start2

        print(ttt)
        picard_iterations.append(int(ttt))

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)

            c1_pred, c2_pred, phi_pred = c1m(x2_re)[1], c2m(x2_re)[1], phim(x2_re)[1]
            c1_exact, c2_exact, phi_exact = ex1_real.c1(x2_re), ex1_real.c2(x2_re), ex1_real.phi(x2_re)

            results['c1_error'][i] = err_calculate.err_L2_2d(c1_pred, c1_exact, wr2, a, b, c, d).item()
            results['c2_error'][i] = err_calculate.err_L2_2d(c2_pred, c2_exact, wr2, a, b, c, d).item()
            results['phi_error'][i] = err_calculate.err_L2_2d(phi_pred, phi_exact, wr2, a, b, c, d).item()
            results['c1_min'][i] = torch.min(c1_pred).item()
            results['c2_min'][i] = torch.min(c2_pred).item()
            results['m1'][i] = err_calculate.int_2d(c1_pred, wr2, a, b, c, d).item()
            results['m2'][i] = err_calculate.int_2d(c2_pred, wr2, a, b, c, d).item()
            results['m1_real'][i] = err_calculate.int_2d(c1_exact, wr2, a, b, c, d).item()
            results['m2_real'][i] = err_calculate.int_2d(c2_exact, wr2, a, b, c, d).item()

            phim_output = phim(x2_re)
            psidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim_output[2]) @ phim.predict.weight.data.T
            psidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim_output[2]) @ phim.predict.weight.data.T

            energy_terms = c1_pred * torch.log(c1_pred) + c2_pred * torch.log(c2_pred) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
            real_energy_terms = c1_exact * torch.log(c1_exact) + c2_exact * torch.log(c2_exact) + eps ** 2 / 2 * (ex1_real.phix(x2_re) ** 2 + ex1_real.phiy(x2_re) ** 2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d).item()
            results['e_real'][i] = err_calculate.int_2d(real_energy_terms, wr2, a, b, c, d).item()

        print(f'c1_L^2 error: {results["c1_error"][-1]}')
        print(f'c2_L^2 error: {results["c2_error"][-1]}')
        print(f'phi_L^2 error: {results["phi_error"][-1]}')

        PNP_plot_st(xt_re, results, k+1)

        xt_re_all[(k+1)*Nre:(k+2)*Nre] = xt_re
        results_all['e_all'][(k+1)*Nre:(k+2)*Nre] = results['e']
        results_all['e_real_all'][(k+1)*Nre:(k+2)*Nre] = results['e_real']
        results_all['m1_all'][(k+1)*Nre:(k+2)*Nre] = results['m1']
        results_all['m2_all'][(k+1)*Nre:(k+2)*Nre] = results['m2']
        results_all['m1_real_all'][(k+1)*Nre:(k+2)*Nre] = results['m1_real']
        results_all['m2_real_all'][(k+1)*Nre:(k+2)*Nre] = results['m2_real']
        results_all['c1_min_all'][(k+1)*Nre:(k+2)*Nre] = results['c1_min']
        results_all['c2_min_all'][(k+1)*Nre:(k+2)*Nre] = results['c2_min']
        results_all['c1_error_all'][(k+1)*Nre:(k+2)*Nre] = results['c1_error']
        results_all['c2_error_all'][(k+1)*Nre:(k+2)*Nre] = results['c2_error']
        results_all['phi_error_all'][(k+1)*Nre:(k+2)*Nre] = results['phi_error']

        t1 = t2

    PNP_plot_st_all(xt_re_all, results_all)

    print(total_time1)
    print(total_time2)
    total_time = total_time1 + total_time2
    print(total_time)

    return {
        'time': xt_re_all.detach().cpu().numpy(),
        'results': {key: np.asarray(value).copy() for key, value in results_all.items()},
        'picard_iterations': picard_iterations,
        'timings': {
            'setup': float(total_time1),
            'picard': float(total_time2),
            'correction': 0.0,
            'total': float(total_time),
        },
    }
