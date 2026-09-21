from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.experiment_io import legacy_output_path, output_path, result_path, save_rows

import ex3_real
import numpy as np
from RNN_ref import err_calculate
from RNN_ref import RNN_funs
from RNN_ref import plot_results
from RNN_ref import fd
import torch
from RNN_ref import sampling_points
import scipy
import time
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float64)


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

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat

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

    ax3.plot(xt_re, results['m1'], label=r'$m_1$', color='green', linestyle='-')
    ax3.plot(xt_re, results['m2'], label=r'$m_2$', color='red', linestyle='--')
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('mass')
    ax3.set_title(r'$m_1$ and $m_2$')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['e'])
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('$E$')
    ax4.set_title('energy')
    ax4.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex3_paper/PNP_r_ex3_paper/SO-RaNN/PNP-st-re-G-ex3_{k}.eps'), bbox_inches='tight')
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

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat

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

    ax3.plot(xt_re, results['m1_all'], label=r'$m_1$', color='green', linestyle='-')
    ax3.plot(xt_re, results['m2_all'], label=r'$m_2$', color='red', linestyle='--')
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('mass')
    ax3.set_title(r'$m_1$ and $m_2$')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['e_all'])
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('$E$')
    ax4.set_title('energy')
    ax4.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_ex3_paper/PNP_r_ex3_paper/SO-RaNN/PNP-st-re-G-all-ex3.eps'), bbox_inches='tight')
    plt.close(fig)


def pnp_ex3_picard_st_G_remass_threshold_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, ite, p1):
    device = torch.device('cpu')
    total_time = 0
    c0 = 10
    ratio = 0.3

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'e_bar': np.zeros(Nre),
        'e': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre)
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt), 'e_bar_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt)
    }

    scaler = {
        'scaler_c1': torch.zeros(Nre * Nt), 'scaler_c2': torch.zeros(Nre * Nt), 'scaler_phi': torch.zeros(Nre * Nt),
    }

    start = time.perf_counter() 

    xr = sampling_points.get_GL_points_3d(n_int)
    wr = sampling_points.get_weights_3d(n_int).reshape([n_int ** 3, 1])
    x = torch.zeros_like(xr)
    x[:, 0:1] = (b - a) / 2 * xr[:, 0:1] + (a + b) / 2
    x[:, 1:2] = (d - c) / 2 * xr[:, 1:2] + (c + d) / 2

    x2_0 = torch.cat((x2, torch.zeros(x2.shape[0], 1)), dim=1)
    with torch.no_grad():
        mass1 = err_calculate.int_2d(ex3_real.c1(x2_0), wr2, a, b, c, d)
        mass2 = err_calculate.int_2d(ex3_real.c2(x2_0), wr2, a, b, c, d)

    stage_records = []
    sav_records = []
    equation_records = []
    previous_stage = {}

    def normalized_lstsq_residual(matrix, rhs, weights):
        """Return the residual of the weighted least-squares system used by the code."""
        residual = matrix @ weights - rhs
        return float(np.linalg.norm(residual) / max(1.0 + np.linalg.norm(rhs), 1e-30))

    def normalized_lstsq_stationarity(matrix, rhs, weights):
        """Measure how closely the coefficients satisfy the least-squares normal equations."""
        residual = matrix @ weights - rhs
        denominator = np.linalg.norm(matrix) * np.linalg.norm(residual)
        return float(np.linalg.norm(matrix.T @ residual) / max(denominator, 1e-30))

    def record_equation_residual(block, strategy, before_components, after_components,
                                 before_stationarity, after_stationarity):
        """Record componentwise and worst-case residuals before and after a re-fit."""
        equation_records.append({
            'block': int(block),
            'strategy': strategy,
            'before_c1': float(before_components.get('c1', np.nan)),
            'after_c1': float(after_components.get('c1', np.nan)),
            'before_c2': float(before_components.get('c2', np.nan)),
            'after_c2': float(after_components.get('c2', np.nan)),
            'before_max': float(max(before_components.values())),
            'after_max': float(max(after_components.values())),
            'before_stationarity_max': float(max(before_stationarity.values())),
            'after_stationarity_max': float(max(after_stationarity.values())),
        })

    def record_sav_sequence(block, time_values, auxiliary_values, correction_factors):
        """Save the blockwise constructed SAV sequence R_1^j and its shifted value."""
        for local_step, (time_value, auxiliary_value, correction_factor) in enumerate(
                zip(time_values, auxiliary_values, correction_factors)):
            sav_records.append({
                'block': int(block),
                'local_step': int(local_step),
                'time': float(time_value),
                'R1': float(auxiliary_value),
                'R1_minus_C0': float(auxiliary_value - c0),
                'xi': float(correction_factor),
            })

    def record_stage(block, stage, time_value, c1_value, c2_value, phi_value,
                     gamma1=1.0, gamma2=1.0, xi=1.0):
        """Record a cumulative correction-stage diagnostic at a block endpoint."""
        c1_value = c1_value.detach()
        c2_value = c2_value.detach()
        phi_value = phi_value.detach()
        m1_value = float(err_calculate.int_2d(c1_value, wr2, a, b, c, d))
        m2_value = float(err_calculate.int_2d(c2_value, wr2, a, b, c, d))
        charge = z1 * m1_value + z2 * m2_value
        charge_scale = abs(z1) * abs(m1_value) + abs(z2) * abs(m2_value)
        integration_scale = (b - a) * (d - c) / 4

        def weighted_norm(value):
            return torch.sqrt(integration_scale * torch.sum(value.square() * wr2)).item()

        prior = previous_stage.get(block)
        if prior is None:
            rel_c1 = rel_c2 = rel_phi = float('nan')
        else:
            rel_c1 = weighted_norm(c1_value - prior[0]) / max(weighted_norm(prior[0]), 1e-30)
            rel_c2 = weighted_norm(c2_value - prior[1]) / max(weighted_norm(prior[1]), 1e-30)
            rel_phi = weighted_norm(phi_value - prior[2]) / max(weighted_norm(prior[2]), 1e-30)

        stage_records.append({
            'block': int(block),
            'time': float(time_value),
            'stage': stage,
            'min_c1': float(torch.min(c1_value)),
            'min_c2': float(torch.min(c2_value)),
            'mass_c1': m1_value,
            'mass_c2': m2_value,
            'mass_defect_c1': abs(m1_value - float(mass1)),
            'mass_defect_c2': abs(m2_value - float(mass2)),
            'relative_mass_defect_c1': abs(m1_value - float(mass1)) / max(abs(float(mass1)), 1e-30),
            'relative_mass_defect_c2': abs(m2_value - float(mass2)) / max(abs(float(mass2)), 1e-30),
            'abs_charge_defect': abs(charge),
            'normalized_charge_defect': abs(charge) / max(charge_scale, 1e-30),
            'gamma1': float(gamma1),
            'gamma2': float(gamma2),
            'xi': float(xi),
            'relative_change_c1': rel_c1,
            'relative_change_c2': rel_c2,
            'relative_change_phi': rel_phi,
        })
        previous_stage[block] = (c1_value.clone(), c2_value.clone(), phi_value.clone())

    def endpoint_fields(time_value):
        endpoint = torch.cat((x2, time_value * torch.ones(x2.shape[0], 1)), dim=1)
        basis1, cutoff1 = c1m(endpoint)[0], c1m(endpoint)[1]
        basis2, cutoff2 = c2m(endpoint)[0], c2m(endpoint)[1]
        return c1m.predict(basis1), c2m.predict(basis2), cutoff1, cutoff2, phim(endpoint)[1]

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

    with torch.no_grad():
        fa1 = np.zeros((N1, 1))
        fa2 = np.zeros((N1, 1))
        fb1 = np.zeros((2*(N2x+N2y), 1))
        fb2 = np.zeros((2*(N2x+N2y), 1))
        fb3 = np.zeros((2*(N2x+N2y), 1))
        fi1 = (ex3_real.c1(xi)).detach().cpu().numpy()
        fi2 = (ex3_real.c2(xi)).detach().cpu().numpy()

    t = torch.linspace(t1, t2, n).reshape(-1, 1)
    E1 = np.zeros((n, m))
    fe1 = np.zeros((n, 1))
    scale_factor = (b - a) * (d - c) / 4
    for j in range(n):
        x2_t = torch.cat((x2, t[j, 0]*torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            output_phi = phim(x2_t)[0]
            E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()

    c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
    c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
    phim_in, phimd1, phimd2, phimdx, phimdy, phimdt, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

    c1md1_b, c2md1_b, phimd1_b = c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
    c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
    c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

    c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
    c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

    phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
    phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

    c1m_i, c2m_i = c1m(xi)[0], c2m(xi)[0]

    diffc1, diffc2, diffphi = 1, 1, 1
    diff_list = {'c1': [], 'c2': [], 'phi': []}
    ttt = 0

    while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
        c1_before, c2_before, phi_before = c1m(x)[1], c2m(x)[1], phim(x)[1]

        # NP
        phimdx_1 = phimdx@phim.predict.weight.data.T
        phimdy_1 = phimdy@phim.predict.weight.data.T
        phimdxx_1 = phimdxx@phim.predict.weight.data.T
        phimdyy_1 = phimdyy@phim.predict.weight.data.T

        A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
        b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
        b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        # P
        c1m_1_in = c1m(xy_inter)[1]
        c2m_1_in = c2m(xy_inter)[1]

        A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
        fa3 = (z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
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

    print(ttt)
    raw1, raw2, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
    record_stage(0, 'raw_picard', t2, raw1, raw2, phi_stage)
    record_stage(0, 'positivity_cutoff', t2, cutoff1, cutoff2, phi_stage)
    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re
    record_stage(0, 'first_mass_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                 phi_stage, s_c1[-1], s_c2[-1])

    # xt_re_np = xt_re.squeeze().numpy()
    # s_c1_np = s_c1.detach().numpy()
    # s_c2_np = s_c2.detach().numpy()

    # s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
    # s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
    # s_c1_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='cubic', fill_value='extrapolate')
    # s_c2_cubic_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='cubic', fill_value='extrapolate')

    # s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    # s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # s_c1_in_0 = torch.from_numpy(s_c1_cubic_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    # s_c2_in_0 = torch.from_numpy(s_c2_cubic_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)


    # c1m_1_in = s_c1_in_0 * c1m(xy_inter)[1]
    # c2m_1_in = s_c2_in_0 * c2m(xy_inter)[1]

    # A3 = -eps * (phimdxx + phimdyy).detach().cpu().numpy()
    # fa3 = (z1 * c1m_1_in + z2 * c2m_1_in).detach().cpu().numpy()
    # B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()
    #
    # b_P = np.vstack((fa3, p1 * fb3, fe1))
    # phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)


    s_phi = torch.ones(Nre) 
    phid1_int_0 = phim(x2_0)[2]
    phidx_int_0 = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_int_0) @ phim.predict.weight.data.T
    phidy_int_0 = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_int_0) @ phim.predict.weight.data.T
    r_0 = err_calculate.int_2d(ex3_real.c1(x2_0)*torch.log(ex3_real.c1(x2_0)) + ex3_real.c2(x2_0)*torch.log(ex3_real.c2(x2_0)) + eps**2/2*(phidx_int_0**2 + phidy_int_0**2), wr2, a, b, c, d) + c0
    results['e_bar'][0] = r_0
    ss = s/(Nre - 1)
    for i in range(Nre - 1):
        x2_re = torch.cat((x2, xt_re[i+1] * torch.ones(x2.shape[0], 1)), dim=1)
        c1m_re = s_c1[i+1] * c1m(x2_re)[1]
        c2m_re = s_c2[i+1] * c2m(x2_re)[1]
        c1md1_re = c1m(x2_re)[2]
        c2md1_re = c2m(x2_re)[2]
        phid1_re = phim(x2_re)[2]
        c1dx_re = s_c1[i+1] * ((c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_re)@c1m.predict.weight.data.T)
        c1dy_re = s_c1[i+1] * ((c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re)@c1m.predict.weight.data.T)
        c2dx_re = s_c2[i+1] * ((c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_re)@c2m.predict.weight.data.T)
        c2dy_re = s_c2[i+1] * ((c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re)@c2m.predict.weight.data.T)
        phidx_re = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_re)@phim.predict.weight.data.T
        phidy_re = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_re)@phim.predict.weight.data.T

        E_n1 = err_calculate.int_2d(c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + eps ** 2 / 2 * (phidx_re ** 2 + phidy_re ** 2), wr2, a, b, c, d) + c0
        Edt = err_calculate.int_2d(D1*c1m_re*((c1dx_re/c1m_re + z1*phidx_re)**2 + (c1dy_re/c1m_re + z1*phidy_re)**2) + D2*c2m_re*((c2dx_re/c2m_re + z2*phidx_re)**2 + (c2dy_re/c2m_re + z2*phidy_re)**2), wr2, a, b, c, d)

        results['e_bar'][i + 1] = (results['e_bar'][i])/(1 + ss*Edt/E_n1)
        s_phi[i+1] = (results['e_bar'][i + 1])/E_n1
        if abs(s_phi[i+1] - 1) > ratio:
            s_phi[i+1] = 1

    record_sav_sequence(0, xt_re, results['e_bar'], s_phi)

    record_stage(0, 'sav_potential_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                 s_phi[-1] * phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])


    xt_re_np = xt_re.squeeze().numpy()
    s_phi_np = s_phi.detach().numpy()
    s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
    s_phi_in_0 = torch.from_numpy(s_phi_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # NP
    phimdx_1 = s_phi_in_0 * phimdx @ phim.predict.weight.data.T
    phimdy_1 = s_phi_in_0 * phimdy @ phim.predict.weight.data.T
    phimdxx_1 = s_phi_in_0 * phimdxx @ phim.predict.weight.data.T
    phimdyy_1 = s_phi_in_0 * phimdyy @ phim.predict.weight.data.T

    A1 = (c1mdt - D1 * (c1mdxx + c1mdyy) - D1 * z1 * (c1mdx * phimdx_1 + c1mdy * phimdy_1 + c1m_in * (phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
    B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
    I1 = (c1m_i).detach().cpu().numpy()

    A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
    b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
    c1_weights_before = c1m.predict.weight.data.T.detach().cpu().numpy().copy()
    np1_residual_before = normalized_lstsq_residual(A_NP1, b_NP1, c1_weights_before)
    np1_stationarity_before = normalized_lstsq_stationarity(A_NP1, b_NP1, c1_weights_before)
    wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
    c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)
    np1_residual_after = normalized_lstsq_residual(A_NP1, b_NP1, wu_NP1[0:m])
    np1_stationarity_after = normalized_lstsq_stationarity(A_NP1, b_NP1, wu_NP1[0:m])

    A2 = (c2mdt - D2 * (c2mdxx + c2mdyy) - D2 * z2 * (c2mdx * phimdx_1 + c2mdy * phimdy_1 + c2m_in * (phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
    B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
    I2 = (c2m_i).detach().cpu().numpy()

    A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
    b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
    c2_weights_before = c2m.predict.weight.data.T.detach().cpu().numpy().copy()
    np2_residual_before = normalized_lstsq_residual(A_NP2, b_NP2, c2_weights_before)
    np2_stationarity_before = normalized_lstsq_stationarity(A_NP2, b_NP2, c2_weights_before)
    wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
    c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)
    np2_residual_after = normalized_lstsq_residual(A_NP2, b_NP2, wu_NP2[0:m])
    np2_stationarity_after = normalized_lstsq_stationarity(A_NP2, b_NP2, wu_NP2[0:m])
    record_equation_residual(
        0,
        'additional_np_solve',
        {'c1': np1_residual_before, 'c2': np2_residual_before},
        {'c1': np1_residual_after, 'c2': np2_residual_after},
        {'c1': np1_stationarity_before, 'c2': np2_stationarity_before},
        {'c1': np1_stationarity_after, 'c2': np2_stationarity_after},
    )

    raw1, raw2, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
    record_stage(0, 'additional_np_raw', t2, raw1, raw2, s_phi[-1] * phi_stage, xi=s_phi[-1])
    record_stage(0, 'additional_np_cutoff', t2, cutoff1, cutoff2, s_phi[-1] * phi_stage, xi=s_phi[-1])

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re
    record_stage(0, 'final_mass_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                 s_phi[-1] * phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])

    xt_re_np = xt_re.squeeze().numpy()
    s_c1_np = s_c1.squeeze().detach().numpy()
    s_c2_np = s_c2.squeeze().detach().numpy()

    s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
    s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
    s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # P
    c1m_1_in = s_c1_in_0 * c1m(xy_inter)[1]
    c2m_1_in = s_c2_in_0 * c2m(xy_inter)[1]

    A3 = -eps**2 * (phimdxx + phimdyy).detach().cpu().numpy()
    fa3 = (z1 * c1m_1_in + z2 * c2m_1_in).detach().cpu().numpy()
    B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

    A_P = np.vstack((A3, p1 * B3, E1)) 
    b_P = np.vstack((fa3, p1 * fb3, fe1))
    s_phi_boundary = torch.from_numpy(
        s_phi_linear_0(xb[:, 2:3].squeeze().numpy())
    ).reshape(-1, 1).numpy()
    s_phi_gauge = torch.from_numpy(
        s_phi_linear_0(t.squeeze().numpy())
    ).reshape(-1, 1).numpy()
    A_P_before = np.vstack((
        s_phi_in_0.detach().cpu().numpy() * A3,
        p1 * s_phi_boundary * B3,
        s_phi_gauge * E1,
    ))
    phi_weights_before = phim.predict.weight.data.T.detach().cpu().numpy().copy()
    poisson_residual_before = normalized_lstsq_residual(A_P_before, b_P, phi_weights_before)
    poisson_stationarity_before = normalized_lstsq_stationarity(A_P_before, b_P, phi_weights_before)
    wu_P = np.linalg.lstsq(A_P, b_P)[0] 
    phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)
    poisson_residual_after = normalized_lstsq_residual(A_P, b_P, wu_P[0:m])
    poisson_stationarity_after = normalized_lstsq_stationarity(A_P, b_P, wu_P[0:m])
    record_equation_residual(
        0,
        'final_poisson_fit',
        {'poisson': poisson_residual_before},
        {'poisson': poisson_residual_after},
        {'poisson': poisson_stationarity_before},
        {'poisson': poisson_stationarity_after},
    )
    _, _, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
    record_stage(0, 'final_poisson_fit', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                 phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])

    torch.save(c1m, legacy_output_path('PNP_code/model_ex3/c1m_0.pth'))
    torch.save(c2m, legacy_output_path('PNP_code/model_ex3/c2m_0.pth'))
    torch.save(phim, legacy_output_path('PNP_code/model_ex3/phim_0.pth'))
    scaler['scaler_c1'][0:Nre] = s_c1
    scaler['scaler_c2'][0:Nre] = s_c2
    scaler['scaler_phi'][0:Nre] = s_phi

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        results['c1_min'][i] = torch.min(s_c1[i] * c1m(x2_re)[1])
        results['c2_min'][i] = torch.min(s_c2[i] * c2m(x2_re)[1])
        results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m(x2_re)[1], wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m(x2_re)[1], wr2, a, b, c, d)
        psidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim(x2_re)[2])@phim.predict.weight.data.T
        psidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim(x2_re)[2])@phim.predict.weight.data.T
        energy_terms = s_c1[i]*c1m(x2_re)[1] * torch.log(s_c1[i]*c1m(x2_re)[1]) + s_c2[i]*c2m(x2_re)[1] * torch.log(s_c2[i]*c2m(x2_re)[1]) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['e_bar_all'][0:Nre] = results['e_bar']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']

    PNP_plot_st_re_G_benchmark(xt_re, results, 0)

    t1 = t2

    for k in range(Nt-1):
        t2 = t1 + s
        x2_0 = torch.cat((x2, t1*torch.ones(x2.shape[0], 1)), dim=1)

        results = {
            'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'e_bar': np.zeros(Nre),
            'e': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre)
        }

        phidx = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        phidy = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        r_n = err_calculate.int_2d(s_c1[-1]*c1m(x2_0)[1] * torch.log(s_c1[-1]*c1m(x2_0)[1]) + s_c2[-1]*c2m(x2_0)[1] * torch.log(s_c2[-1]*c2m(x2_0)[1]), wr2, a, b, c, d) + c0
        E0_n = err_calculate.int_2d(eps ** 2 / 2 * (phidx ** 2 + phidy ** 2), wr2, a, b, c, d)
        results['e_bar'][0] = r_n + E0_n

        start = time.perf_counter()

        xy_inter = sampling_points.get_interior_points_rec_3d(N1, a, b, c, d, t1, t2)
        xb = sampling_points.get_boundary_points_rec_3d_t(N2x, N2y, a, b, c, d, t1, t2)
        xi = sampling_points.get_initial_points_rec_3d_t(N3, a, b, c, d, t1)
        xt_re = torch.linspace(t1, t2, Nre)
        x[:, 2:3] = (t2 - t1) / 2 * xr[:, 2:3] + (t1 + t2) / 2

        with torch.no_grad():
            fa1 = np.zeros((N1, 1))
            fa2 = np.zeros((N1, 1))
            fb1 = np.zeros((2 * (N2x + N2y), 1))
            fb2 = np.zeros((2 * (N2x + N2y), 1))
            fb3 = np.zeros((2 * (N2x + N2y), 1))
            fi1 = (s_c1[-1] * c1m(xi)[1]).detach().cpu().numpy()
            fi2 = (s_c2[-1] * c2m(xi)[1]).detach().cpu().numpy()

        t = torch.linspace(t1, t2, n).reshape(-1, 1)
        E1 = np.zeros((n, m))
        fe1 = np.zeros((n, 1))
        scale_factor = (b - a) * (d - c) / 4
        for j in range(n):
            x2_t = torch.cat((x2, t[j, 0] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                output_phi = phim(x2_t)[0]
                E1[j] = (scale_factor * torch.sum(output_phi * wr2, dim=0, keepdim=True)).cpu().numpy()

        c1m_in, c1md1, c1md2, c1mdx, c1mdy, c1mdt, c1mdxx, c1mdyy = RNN_funs.compute_nn_derivatives_1d(c1m, m, xy_inter)
        c2m_in, c2md1, c2md2, c2mdx, c2mdy, c2mdt, c2mdxx, c2mdyy = RNN_funs.compute_nn_derivatives_1d(c2m, m, xy_inter)
        phim_in, phimd1, phimd2, phimdx, phimdy, phimdt, phimdxx, phimdyy = RNN_funs.compute_nn_derivatives_1d(phim, m, xy_inter)

        c1md1_b, c2md1_b, phimd1_b = c1m(xb)[2], c2m(xb)[2], phim(xb)[2]
        c1mdx_b = c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_b
        c1mdy_b = c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_b

        c2mdx_b = c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_b
        c2mdy_b = c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_b

        phimdx_b = phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phimd1_b
        phimdy_b = phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phimd1_b

        c1m_i, c2m_i = c1m(xi)[0], c2m(xi)[0]

        diffc1, diffc2, diffphi = 1, 1, 1
        diff_list = {'c1': [], 'c2': [], 'phi': []}
        ttt = 0

        while (diffc1 > threshold or diffc2 > threshold or diffphi > threshold) and ttt<ite:
            c1_before = c1m(x)[1]
            c2_before = c2m(x)[1]
            phi_before = phim(x)[1]

            # NP
            phimdx_1 = phimdx@phim.predict.weight.data.T
            phimdy_1 = phimdy@phim.predict.weight.data.T
            phimdxx_1 = phimdxx@phim.predict.weight.data.T
            phimdyy_1 = phimdyy@phim.predict.weight.data.T

            A1 = (c1mdt - D1*(c1mdxx + c1mdyy) - D1*z1*(c1mdx*phimdx_1 + c1mdy*phimdy_1 + c1m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            B1 = (c1mdx_b*n1 + c1mdy_b*n2).detach().cpu().numpy()
            I1 = (c1m_i).detach().cpu().numpy()

            A_NP1 = np.vstack((A1, p1*B1, p1*I1)) 
            b_NP1 = np.vstack((fa1, p1*fb1, p1*fi1))
            wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
            c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

            A2 = (c2mdt - D2*(c2mdxx + c2mdyy) - D2*z2*(c2mdx*phimdx_1 + c2mdy*phimdy_1 + c2m_in*(phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
            B2 = (c2mdx_b*n1 + c2mdy_b*n2).detach().cpu().numpy()
            I2 = (c2m_i).detach().cpu().numpy()

            A_NP2 = np.vstack((A2, p1*B2, p1*I2)) 
            b_NP2 = np.vstack((fa2, p1*fb2, p1*fi2))
            wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
            c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

            # P
            c1m_1_in = c1m(xy_inter)[1]
            c2m_1_in = c2m(xy_inter)[1]

            A3 = -eps**2*(phimdxx + phimdyy).detach().cpu().numpy()
            fa3 = (z1*c1m_1_in + z2*c2m_1_in).detach().cpu().numpy()
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

        print(ttt)
        raw1, raw2, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
        record_stage(k + 1, 'raw_picard', t2, raw1, raw2, phi_stage)
        record_stage(k + 1, 'positivity_cutoff', t2, cutoff1, cutoff2, phi_stage)
        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)
        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)
        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re
        record_stage(k + 1, 'first_mass_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                     phi_stage, s_c1[-1], s_c2[-1])


        s_phi = torch.ones(Nre) 
        ss = s/(Nre - 1)
        for i in range(Nre - 1):
            x2_re = torch.cat((x2, xt_re[i+1] * torch.ones(x2.shape[0], 1)), dim=1)
            c1m_re = s_c1[i+1] * c1m(x2_re)[1]
            c2m_re = s_c2[i+1] * c2m(x2_re)[1]
            c1md1_re = c1m(x2_re)[2]
            c2md1_re = c2m(x2_re)[2]
            phid1_re = phim(x2_re)[2]
            c1dx_re = s_c1[i+1] * ((c1m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c1md1_re)@c1m.predict.weight.data.T)
            c1dy_re = s_c1[i+1] * ((c1m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re)@c1m.predict.weight.data.T)
            c2dx_re = s_c2[i+1] * ((c2m.hidden.weight.data[:, 0:1].reshape([-1, m]) * c2md1_re)@c2m.predict.weight.data.T)
            c2dy_re = s_c2[i+1] * ((c2m.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re)@c2m.predict.weight.data.T)
            phidx_re = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_re)@phim.predict.weight.data.T
            phidy_re = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_re)@phim.predict.weight.data.T

            E_n1 = err_calculate.int_2d(c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + eps ** 2 / 2 * (phidx_re ** 2 + phidy_re ** 2), wr2, a, b, c, d) + c0
            Edt = err_calculate.int_2d(D1*c1m_re*((c1dx_re/c1m_re + z1*phidx_re)**2 + (c1dy_re/c1m_re + z1*phidy_re)**2) + D2*c2m_re*((c2dx_re/c2m_re + z2*phidx_re)**2 + (c2dy_re/c2m_re + z2*phidy_re)**2), wr2, a, b, c, d)
            results['e_bar'][i + 1] = (results['e_bar'][i])/(1 + ss*Edt/E_n1)
            s_phi[i+1] = (results['e_bar'][i + 1])/E_n1
            if abs(s_phi[i + 1] - 1) > ratio:
                s_phi[i + 1] = 1

        record_sav_sequence(k + 1, xt_re, results['e_bar'], s_phi)

        record_stage(k + 1, 'sav_potential_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                     s_phi[-1] * phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])


        xt_re_np = xt_re.squeeze().numpy()
        s_phi_np = s_phi.detach().numpy()
        s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
        s_phi_in_0 = torch.from_numpy(s_phi_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

        # NP
        phimdx_1 = s_phi_in_0 * phimdx @ phim.predict.weight.data.T
        phimdy_1 = s_phi_in_0 * phimdy @ phim.predict.weight.data.T
        phimdxx_1 = s_phi_in_0 * phimdxx @ phim.predict.weight.data.T
        phimdyy_1 = s_phi_in_0 * phimdyy @ phim.predict.weight.data.T

        A1 = (c1mdt - D1 * (c1mdxx + c1mdyy) - D1 * z1 * (c1mdx * phimdx_1 + c1mdy * phimdy_1 + c1m_in * (phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
        b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
        c1_weights_before = c1m.predict.weight.data.T.detach().cpu().numpy().copy()
        np1_residual_before = normalized_lstsq_residual(A_NP1, b_NP1, c1_weights_before)
        np1_stationarity_before = normalized_lstsq_stationarity(A_NP1, b_NP1, c1_weights_before)
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)
        np1_residual_after = normalized_lstsq_residual(A_NP1, b_NP1, wu_NP1[0:m])
        np1_stationarity_after = normalized_lstsq_stationarity(A_NP1, b_NP1, wu_NP1[0:m])

        A2 = (c2mdt - D2 * (c2mdxx + c2mdyy) - D2 * z2 * (c2mdx * phimdx_1 + c2mdy * phimdy_1 + c2m_in * (phimdxx_1 + phimdyy_1))).detach().cpu().numpy()
        B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
        b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
        c2_weights_before = c2m.predict.weight.data.T.detach().cpu().numpy().copy()
        np2_residual_before = normalized_lstsq_residual(A_NP2, b_NP2, c2_weights_before)
        np2_stationarity_before = normalized_lstsq_stationarity(A_NP2, b_NP2, c2_weights_before)
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)
        np2_residual_after = normalized_lstsq_residual(A_NP2, b_NP2, wu_NP2[0:m])
        np2_stationarity_after = normalized_lstsq_stationarity(A_NP2, b_NP2, wu_NP2[0:m])
        record_equation_residual(
            k + 1,
            'additional_np_solve',
            {'c1': np1_residual_before, 'c2': np2_residual_before},
            {'c1': np1_residual_after, 'c2': np2_residual_after},
            {'c1': np1_stationarity_before, 'c2': np2_stationarity_before},
            {'c1': np1_stationarity_after, 'c2': np2_stationarity_after},
        )

        raw1, raw2, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
        record_stage(k + 1, 'additional_np_raw', t2, raw1, raw2, s_phi[-1] * phi_stage, xi=s_phi[-1])
        record_stage(k + 1, 'additional_np_cutoff', t2, cutoff1, cutoff2, s_phi[-1] * phi_stage, xi=s_phi[-1])

        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re
        record_stage(k + 1, 'final_mass_scaling', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                     s_phi[-1] * phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])

        xt_re_np = xt_re.squeeze().numpy()
        s_c1_np = s_c1.squeeze().detach().numpy()
        s_c2_np = s_c2.squeeze().detach().numpy()

        s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
        s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
        s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

        # P
        c1m_1_in = s_c1_in_0 * c1m(xy_inter)[1]
        c2m_1_in = s_c2_in_0 * c2m(xy_inter)[1]

        A3 = -eps**2 * (phimdxx + phimdyy).detach().cpu().numpy()
        fa3 = (z1 * c1m_1_in + z2 * c2m_1_in).detach().cpu().numpy()
        B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1 * B3, E1)) 
        b_P = np.vstack((fa3, p1 * fb3, fe1))
        s_phi_boundary = torch.from_numpy(
            s_phi_linear_0(xb[:, 2:3].squeeze().numpy())
        ).reshape(-1, 1).numpy()
        s_phi_gauge = torch.from_numpy(
            s_phi_linear_0(t.squeeze().numpy())
        ).reshape(-1, 1).numpy()
        A_P_before = np.vstack((
            s_phi_in_0.detach().cpu().numpy() * A3,
            p1 * s_phi_boundary * B3,
            s_phi_gauge * E1,
        ))
        phi_weights_before = phim.predict.weight.data.T.detach().cpu().numpy().copy()
        poisson_residual_before = normalized_lstsq_residual(A_P_before, b_P, phi_weights_before)
        poisson_stationarity_before = normalized_lstsq_stationarity(A_P_before, b_P, phi_weights_before)
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)
        poisson_residual_after = normalized_lstsq_residual(A_P, b_P, wu_P[0:m])
        poisson_stationarity_after = normalized_lstsq_stationarity(A_P, b_P, wu_P[0:m])
        record_equation_residual(
            k + 1,
            'final_poisson_fit',
            {'poisson': poisson_residual_before},
            {'poisson': poisson_residual_after},
            {'poisson': poisson_stationarity_before},
            {'poisson': poisson_stationarity_after},
        )
        _, _, cutoff1, cutoff2, phi_stage = endpoint_fields(t2)
        record_stage(k + 1, 'final_poisson_fit', t2, s_c1[-1] * cutoff1, s_c2[-1] * cutoff2,
                     phi_stage, s_c1[-1], s_c2[-1], s_phi[-1])

        end = time.perf_counter()
        total_time += (end - start)

        torch.save(c1m, legacy_output_path(f'PNP_code/model_ex3/c1m_{k+1}.pth'))
        torch.save(c2m, legacy_output_path(f'PNP_code/model_ex3/c2m_{k+1}.pth'))
        torch.save(phim, legacy_output_path(f'PNP_code/model_ex3/phim_{k+1}.pth'))
        scaler['scaler_phi'][(k+1)*Nre:(k+2)*Nre] = s_phi
        scaler['scaler_c1'][(k+1)*Nre:(k+2)*Nre] = s_c1
        scaler['scaler_c2'][(k+1)*Nre:(k+2)*Nre] = s_c2

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            results['c1_min'][i] = torch.min(s_c1[i] * c1m(x2_re)[1])
            results['c2_min'][i] = torch.min(s_c2[i] * c2m(x2_re)[1])
            results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m(x2_re)[1], wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m(x2_re)[1], wr2, a, b, c, d)
            psidx = ((phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim(x2_re)[2])@phim.predict.weight.data.T)
            psidy = ((phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim(x2_re)[2])@phim.predict.weight.data.T)
            energy_terms = s_c1[i]*c1m(x2_re)[1] * torch.log(s_c1[i]*c1m(x2_re)[1]) + s_c2[i]*c2m(x2_re)[1] * torch.log(s_c2[i]*c2m(x2_re)[1]) + eps ** 2 / 2 * (psidx ** 2 + psidy ** 2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

        xt_re_all[(k+1)*Nre:(k+2)*Nre]  = xt_re
        results_all['e_all'][(k+1)*Nre:(k+2)*Nre] = results['e']
        results_all['e_bar_all'][(k+1)*Nre:(k+2)*Nre] = results['e_bar']
        results_all['m1_all'][(k+1)*Nre:(k+2)*Nre] = results['m1']
        results_all['m2_all'][(k+1)*Nre:(k+2)*Nre] = results['m2']
        results_all['c1_min_all'][(k+1)*Nre:(k+2)*Nre] = results['c1_min']
        results_all['c2_min_all'][(k+1)*Nre:(k+2)*Nre] = results['c2_min']

        PNP_plot_st_re_G_benchmark(xt_re, results, k + 1)

        t1 = t2

    np.savetxt(result_path('PNP_code', 'example4_3', 'scaler_phi.txt'), scaler['scaler_phi'].detach().numpy(), fmt='%.6f')

    PNP_plot_st_re_G_all_benchmark(xt_re_all, results_all)

    print(total_time)

    save_rows(output_path('diagnostics', 'example4_3_stage_ablation.csv'), stage_records)
    save_rows(output_path('diagnostics', 'example4_3_sav_energy.csv'), sav_records)
    save_rows(output_path('diagnostics', 'example4_3_equation_residuals.csv'), equation_records)

    return scaler
