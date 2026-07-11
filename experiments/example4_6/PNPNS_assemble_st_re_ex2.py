from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.paths import legacy_output_path

import ex2_real
import numpy as np
from RNN_ref import err_calculate
from RNN_ref import fd
import torch
from RNN_ref import sampling_points
import time
from RNN_ref import RNN_funs
import scipy

def PNPNS_plot_st_re_benchmark(xt_re, results, k):
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
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

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

    ax3.plot(xt_re, results['m1'])
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('mass')
    ax3.set_title(r'$m_1$')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['m2'])
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('mass')
    ax4.set_title(r'$m_2$')
    ax4.grid(True, alpha=0.3)

    ax5.plot(xt_re, results['Divergence_free'])
    ax5.set_xlabel('$t$')
    ax5.set_ylabel('$L^2$ norm')
    ax5.set_title(r'$\Vert \nabla \cdot \mathbf{u} \Vert$')
    ax5.grid(True, alpha=0.3)

    ax6.plot(xt_re, results['e'])
    ax6.set_xlabel('$t$')
    ax6.set_ylabel('$E$')
    ax6.set_title('energy')
    ax6.grid(True, alpha=0.3)


    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex2/PNPNS-st-re-ex2_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNPNS_plot_st_re_all_benchmark(xt_re, results):
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
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

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

    ax3.plot(xt_re, results['m1_all'])
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('mass')
    ax3.set_title(r'$m_1$')
    ax3.grid(True, alpha=0.3)

    ax4.plot(xt_re, results['m2_all'])
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('mass')
    ax4.set_title(r'$m_2$')
    ax4.grid(True, alpha=0.3)

    ax5.plot(xt_re, results['Divergence_free_all'])
    ax5.set_xlabel('$t$')
    ax5.set_ylabel('$L^2$ norm')
    ax5.set_title(r'$\Vert \nabla \cdot \mathbf{u} \Vert$')
    ax5.grid(True, alpha=0.3)

    ax6.plot(xt_re, results['e_all'])
    ax6.set_xlabel('$t$')
    ax6.set_ylabel('$E$')
    ax6.set_title('energy')
    ax6.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNPNS_code/PNPNS_results_ex2/PNPNS-st-re-all-ex2.eps'), bbox_inches='tight')
    plt.close(fig)

def pnpns_ex2_picard_st_decoupled_remass_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, um, pm, a, b, c, d, t1, D1, D2, z1, z2, eps, nv, threshold, ite, p1):
    device = 'cpu'
    total_time = 0
    c0 = 10

    results = {
        'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'e': np.zeros(Nre), 'Divergence_free': np.zeros(Nre),
        'e_real': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre), 'e_bar': np.zeros(Nre),
    }

    xt_re_all = torch.zeros(Nre * Nt)

    results_all = {
        'm1_all': np.zeros(Nre * Nt), 'm2_all': np.zeros(Nre * Nt), 'Divergence_free_all': np.zeros(Nre * Nt),
        'e_all': np.zeros(Nre * Nt), 'c1_min_all': np.zeros(Nre * Nt), 'c2_min_all': np.zeros(Nre * Nt),
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
        mass1 = err_calculate.int_2d(ex2_real.c1(x2_0), wr2, a, b, c, d)
        mass2 = err_calculate.int_2d(ex2_real.c2(x2_0), wr2, a, b, c, d)

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

    fa1 = np.zeros((N1, 1))
    fa2 = np.zeros((N1, 1))
    fb1 = np.zeros((2 * (N2x + N2y), 1))
    fb2 = np.zeros((2 * (N2x + N2y), 1))
    fb3 = np.zeros((2 * (N2x + N2y), 1))
    fb4 = np.zeros((2 * (N2x + N2y), 1))
    fb5 = np.zeros((2 * (N2x + N2y), 1))
    fi1 = (ex2_real.c1(xi)).detach().cpu().numpy()
    fi2 = (ex2_real.c2(xi)).detach().cpu().numpy()
    fi3 = (ex2_real.u1(xi)).detach().cpu().numpy()
    fi4 = (ex2_real.u2(xi)).detach().cpu().numpy()

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
        fa3 = (z1*c1m_in_1 + z2*c2m_in_1).detach().cpu().numpy()

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
        fc1 = (- phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

        C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
        C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
        fc2 = (- phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

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

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re

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

    end = time.perf_counter()
    total_time += (end - start)

    s_phi = torch.ones(Nre) 
    phid1_0 = phim(x2_0)[2]
    phidx_0 = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
    phidy_0 = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
    r_0 = err_calculate.int_2d(ex2_real.c1(x2_0)*torch.log(ex2_real.c1(x2_0)) + ex2_real.c2(x2_0)*torch.log(ex2_real.c2(x2_0)) + eps**2/2*(phidx_0**2 + phidy_0**2) + 0.5*(ex2_real.u1(x2_0)**2 + ex2_real.u2(x2_0)**2), wr2, a, b, c, d) + c0
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
        u1m_re = um(x2_re)[3]
        u2m_re = um(x2_re)[4]
        u1dx_re = ((um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re)@um.predict.weight.data.T)
        u1dy_re = ((um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * c1md1_re)@um.predict.weight.data.T)
        u2dx_re = ((-um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * c2md1_re)@um.predict.weight.data.T)
        u2dy_re = ((-um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re)@um.predict.weight.data.T)

        E_n1 = err_calculate.int_2d(c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + eps ** 2 / 2 * (phidx_re ** 2 + phidy_re ** 2) + 0.5*(u1m_re**2 + u2m_re**2), wr2, a, b, c, d) + c0
        Edt = err_calculate.int_2d(D1*c1m_re*((c1dx_re/c1m_re + z1*phidx_re)**2 + (c1dy_re/c1m_re + z1*phidy_re)**2) + D2*c2m_re*((c2dx_re/c2m_re + z2*phidx_re)**2 + (c2dy_re/c2m_re + z2*phidy_re)**2) + nv*(u1dx_re**2 + u1dy_re**2 + u2dx_re**2 + u2dy_re**2), wr2, a, b, c, d)
        results['e_bar'][i + 1] = results['e_bar'][i]/(1 + ss*Edt/E_n1)
        s_phi[i+1] = results['e_bar'][i + 1]/E_n1

    xt_re_np = xt_re.squeeze().numpy()
    s_phi_np = s_phi.detach().numpy()
    s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
    s_phi_in_0 = torch.from_numpy(s_phi_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # NP
    u1m_in_1 = um(xy_inter)[3]
    u2m_in_1 = um(xy_inter)[4]
    phimdx_in_1 = s_phi_in_0 * phimdx_in @ phim.predict.weight.data.T
    phimdy_in_1 = s_phi_in_0 * phimdy_in @ phim.predict.weight.data.T
    phimdxx_in_1 = s_phi_in_0 * phimdxx_in @ phim.predict.weight.data.T
    phimdyy_in_1 = s_phi_in_0 * phimdyy_in @ phim.predict.weight.data.T

    A1 = (c1mdt_in + (u1m_in_1 * c1mdx_in + u2m_in_1 * c1mdy_in) - D1 * (c1mdxx_in + c1mdyy_in + z1 * (c1m_in * (
                phimdxx_in_1 + phimdyy_in_1) + c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1))).detach().cpu().numpy()
    B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
    I1 = (c1m_i).detach().cpu().numpy()

    A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
    b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
    wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
    c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

    A2 = (c2mdt_in + (u1m_in_1 * c2mdx_in + u2m_in_1 * c2mdy_in) - D2 * (c2mdxx_in + c2mdyy_in + z2 * (c2m_in * (
                phimdxx_in_1 + phimdyy_in_1) + c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1))).detach().cpu().numpy()
    B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
    I2 = (c2m_i).detach().cpu().numpy()

    A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
    b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
    wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
    c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

    mass1_re = torch.zeros(Nre)
    mass2_re = torch.zeros(Nre)

    for i in range(Nre):
        x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        with torch.no_grad():
            mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
            mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

    s_c1 = mass1 / mass1_re
    s_c2 = mass2 / mass2_re

    xt_re_np = xt_re.squeeze().numpy()
    s_c1_np = s_c1.squeeze().detach().numpy()
    s_c2_np = s_c2.squeeze().detach().numpy()

    s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
    s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
    s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
    s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

    # P
    c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
    c2m_in_1 = s_c2_in_0 * c2m(xy_inter)[1]

    A3 = -eps * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
    fa3 = (z1 * c1m_in_1 + z2 * c2m_in_1).detach().cpu().numpy()

    B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

    A_P = np.vstack((A3, p1 * B3, E1)) 
    b_P = np.vstack((fa3, p1 * fb3, fe1))
    wu_P = np.linalg.lstsq(A_P, b_P)[0] 
    phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

    # NS
    phimdx_1 = phimdx_in @ phim.predict.weight.data.T
    phimdy_1 = phimdy_in @ phim.predict.weight.data.T

    C1[:, 0:m] = (u1mdt_in + u1m_in_1 * u1mdx_in + u2m_in_1 * u1mdy_in - nv * (
                u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
    C1[:, m:2 * m] = (pmdx_in).detach().cpu().numpy()
    fc1 = (- phimdx_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()

    C2[:, 0:m] = (u2mdt_in + u1m_in_1 * u2mdx_in + u2m_in_1 * u2mdy_in - nv * (
                u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
    C2[:, m:2 * m] = (pmdy_in).detach().cpu().numpy()
    fc2 = (- phimdy_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()

    B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
    B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
    I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
    I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

    A_NS = np.vstack((C1, C2, p1 * B4, p1 * B5, p1 * I3, p1 * I4, E2)) 
    b_NS = np.vstack((fc1, fc2, p1 * fb4, p1 * fb5, p1 * fi3, p1 * fi4, fe2))
    # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
    wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
    um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
    pm.predict.weight.data = torch.from_numpy(wu_NS[m:2 * m]).T.to(device)

    end = time.perf_counter()
    total_time += (end - start)

    torch.save(c1m, legacy_output_path('PNPNS_code/model_ex2/c1m_0.pth'))
    torch.save(c2m, legacy_output_path('PNPNS_code/model_ex2/c2m_0.pth'))
    # torch.save(phim, legacy_output_path('PNPNS_code/model_ex2/phim_0.pth'))
    torch.save(um.state_dict(), legacy_output_path('PNPNS_code/model_ex2/um_weights_0.pth'))
    # torch.save(pm, legacy_output_path('PNPNS_code/model_ex2/pm_0.pth'))
    scaler['scaler_c1'][0:Nre] = s_c1
    scaler['scaler_c2'][0:Nre] = s_c2
    scaler['scaler_phi'][0:Nre] = s_phi

    for i in range(Nre):
        x2_plot = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
        x2_plot.requires_grad_(True)
        c1m_plot = s_c1[i] * c1m(x2_plot)[1]
        c2m_plot = s_c2[i] * c2m(x2_plot)[1]
        u1m_plot = um(x2_plot)[3]
        u2m_plot = um(x2_plot)[4]
        results['c1_min'][i] = torch.min(c1m_plot)
        results['c2_min'][i] = torch.min(c2m_plot)
        results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m(x2_plot)[1], wr2, a, b, c, d)
        results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m(x2_plot)[1], wr2, a, b, c, d)
        results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot, u2m_plot, x2_plot, wr2, a, b, c, d)
        phid1_plot = phim(x2_plot)[2]
        phidx_plot = ((phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_plot)@phim.predict.weight.data.T)
        phidy_plot = ((phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_plot)@phim.predict.weight.data.T)
        energy_terms = c1m_plot * torch.log(c1m_plot) + c2m_plot * torch.log(c2m_plot) + eps ** 2 / 2 * (phidx_plot ** 2 + phidy_plot ** 2) + 0.5*(u1m_plot**2 + u2m_plot**2)
        results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

    xt_re_all[0:Nre] = xt_re
    results_all['e_all'][0:Nre] = results['e']
    results_all['m1_all'][0:Nre] = results['m1']
    results_all['m2_all'][0:Nre] = results['m2']
    results_all['c1_min_all'][0:Nre] = results['c1_min']
    results_all['c2_min_all'][0:Nre] = results['c2_min']
    results_all['Divergence_free_all'][0:Nre] = results['Divergence_free']

    PNPNS_plot_st_re_benchmark(xt_re, results, 0)

    t1 = t2

    for k in range(Nt-1):

        t2 = t1 + s

        results = {
            'm1': np.zeros(Nre), 'm2': np.zeros(Nre), 'e': np.zeros(Nre), 'Divergence_free': np.zeros(Nre),
            'e_real': np.zeros(Nre), 'c1_min': np.zeros(Nre), 'c2_min': np.zeros(Nre), 'e_bar': np.zeros(Nre),
        }

        c1m_int_n = s_c1[-1] * c1m(x2_0)[1]
        c2m_int_n = s_c2[-1] * c2m(x2_0)[1]
        phidx_int_n = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        phidy_int_n = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phim(x2_0)[2]) @ phim.predict.weight.data.T
        u1m_int_n = um(x2_0)[3]
        u2m_int_n = um(x2_0)[4]
        r_n = err_calculate.int_2d(c1m_int_n * torch.log(c1m_int_n) + c2m_int_n * torch.log(c2m_int_n) + eps ** 2 / 2 * (phidx_int_n ** 2 + phidy_int_n ** 2) + 0.5*(u1m_int_n**2 + u2m_int_n**2), wr2, a, b, c, d) + c0
        results['e_bar'][0] = r_n

        start = time.perf_counter()

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

        fa1 = np.zeros((N1, 1))
        fa2 = np.zeros((N1, 1))
        fb1 = np.zeros((2 * (N2x + N2y), 1))
        fb2 = np.zeros((2 * (N2x + N2y), 1))
        fb3 = np.zeros((2 * (N2x + N2y), 1))
        fb4 = np.zeros((2 * (N2x + N2y), 1))
        fb5 = np.zeros((2 * (N2x + N2y), 1))
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
            fa3 = (z1*c1m_in_1 + z2*c2m_in_1).detach().cpu().numpy()

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
            fc1 = (- phimdx_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

            C2[:, 0:m] = (u2mdt_in + u1m_in_1*u2mdx_in + u2m_in_1*u2mdy_in - nv*(u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
            C2[:, m:2*m] = (pmdy_in).detach().cpu().numpy()
            fc2 = (- phimdy_1*(z1*c1m_in_1 + z2*c2m_in_1)).detach().cpu().numpy()

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
        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():

                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re

        s_phi = torch.ones(Nre) 
        phid1_0 = phim(x2_0)[2]
        phidx_0 = (phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
        phidy_0 = (phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_0) @ phim.predict.weight.data.T
        r_0 = err_calculate.int_2d(ex2_real.c1(x2_0)*torch.log(ex2_real.c1(x2_0)) + ex2_real.c2(x2_0)*torch.log(ex2_real.c2(x2_0)) + eps**2/2*(phidx_0**2 + phidy_0**2) + 0.5*(ex2_real.u1(x2_0)**2 + ex2_real.u2(x2_0)**2), wr2, a, b, c, d) + c0
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
            u1m_re = um(x2_re)[3]
            u2m_re = um(x2_re)[4]
            u1dx_re = ((um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * c1md1_re)@um.predict.weight.data.T)
            u1dy_re = ((um.hidden.weight.data[:, 1:2].reshape([-1, m]) ** 2 * c1md1_re)@um.predict.weight.data.T)
            u2dx_re = ((-um.hidden.weight.data[:, 0:1].reshape([-1, m]) ** 2 * c2md1_re)@um.predict.weight.data.T)
            u2dy_re = ((-um.hidden.weight.data[:, 0:1].reshape([-1, m]) * um.hidden.weight.data[:, 1:2].reshape([-1, m]) * c2md1_re)@um.predict.weight.data.T)

            E_n1 = err_calculate.int_2d(c1m_re * torch.log(c1m_re) + c2m_re * torch.log(c2m_re) + eps ** 2 / 2 * (phidx_re ** 2 + phidy_re ** 2) + 0.5*(u1m_re**2 + u2m_re**2), wr2, a, b, c, d) + c0
            Edt = err_calculate.int_2d(D1*c1m_re*((c1dx_re/c1m_re + z1*phidx_re)**2 + (c1dy_re/c1m_re + z1*phidy_re)**2) + D2*c2m_re*((c2dx_re/c2m_re + z2*phidx_re)**2 + (c2dy_re/c2m_re + z2*phidy_re)**2) + nv*(u1dx_re**2 + u1dy_re**2 + u2dx_re**2 + u2dy_re**2), wr2, a, b, c, d)
            results['e_bar'][i + 1] = results['e_bar'][i]/(1 + ss*Edt/E_n1)
            s_phi[i+1] = results['e_bar'][i + 1]/E_n1

        xt_re_np = xt_re.squeeze().numpy()
        s_phi_np = s_phi.detach().numpy()
        s_phi_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_phi_np, kind='linear', fill_value='extrapolate')
        s_phi_in_0 = torch.from_numpy(s_phi_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

        # NP
        u1m_in_1 = um(xy_inter)[3]
        u2m_in_1 = um(xy_inter)[4]
        phimdx_in_1 = s_phi_in_0 * phimdx_in @ phim.predict.weight.data.T
        phimdy_in_1 = s_phi_in_0 * phimdy_in @ phim.predict.weight.data.T
        phimdxx_in_1 = s_phi_in_0 * phimdxx_in @ phim.predict.weight.data.T
        phimdyy_in_1 = s_phi_in_0 * phimdyy_in @ phim.predict.weight.data.T

        A1 = (c1mdt_in + (u1m_in_1 * c1mdx_in + u2m_in_1 * c1mdy_in) - D1 * (c1mdxx_in + c1mdyy_in + z1 * (c1m_in * (
                phimdxx_in_1 + phimdyy_in_1) + c1mdx_in * phimdx_in_1 + c1mdy_in * phimdy_in_1))).detach().cpu().numpy()
        B1 = (c1mdx_b * n1 + c1mdy_b * n2).detach().cpu().numpy()
        I1 = (c1m_i).detach().cpu().numpy()

        A_NP1 = np.vstack((A1, p1 * B1, p1 * I1)) 
        b_NP1 = np.vstack((fa1, p1 * fb1, p1 * fi1))
        wu_NP1 = np.linalg.lstsq(A_NP1, b_NP1)[0] 
        c1m.predict.weight.data = torch.from_numpy(wu_NP1[0:m]).T.to(device)

        A2 = (c2mdt_in + (u1m_in_1 * c2mdx_in + u2m_in_1 * c2mdy_in) - D2 * (c2mdxx_in + c2mdyy_in + z2 * (c2m_in * (
                phimdxx_in_1 + phimdyy_in_1) + c2mdx_in * phimdx_in_1 + c2mdy_in * phimdy_in_1))).detach().cpu().numpy()
        B2 = (c2mdx_b * n1 + c2mdy_b * n2).detach().cpu().numpy()
        I2 = (c2m_i).detach().cpu().numpy()

        A_NP2 = np.vstack((A2, p1 * B2, p1 * I2)) 
        b_NP2 = np.vstack((fa2, p1 * fb2, p1 * fi2))
        wu_NP2 = np.linalg.lstsq(A_NP2, b_NP2)[0] 
        c2m.predict.weight.data = torch.from_numpy(wu_NP2[0:m]).T.to(device)

        mass1_re = torch.zeros(Nre)
        mass2_re = torch.zeros(Nre)

        for i in range(Nre):
            x2_re = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            with torch.no_grad():
                mass1_re[i] = err_calculate.int_2d(c1m(x2_re)[1], wr2, a, b, c, d)
                mass2_re[i] = err_calculate.int_2d(c2m(x2_re)[1], wr2, a, b, c, d)

        s_c1 = mass1 / mass1_re
        s_c2 = mass2 / mass2_re

        xt_re_np = xt_re.squeeze().numpy()
        s_c1_np = s_c1.squeeze().detach().numpy()
        s_c2_np = s_c2.squeeze().detach().numpy()

        s_c1_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c1_np, kind='linear', fill_value='extrapolate')
        s_c2_linear_0 = scipy.interpolate.interp1d(xt_re_np, s_c2_np, kind='linear', fill_value='extrapolate')
        s_c1_in_0 = torch.from_numpy(s_c1_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)
        s_c2_in_0 = torch.from_numpy(s_c2_linear_0(xy_inter[:, 2:3].squeeze().numpy())).reshape(-1, 1)

        # P
        c1m_in_1 = s_c1_in_0 * c1m(xy_inter)[1]
        c2m_in_1 = s_c2_in_0 * c2m(xy_inter)[1]

        A3 = -eps * (phimdxx_in + phimdyy_in).detach().cpu().numpy()
        fa3 = (z1 * c1m_in_1 + z2 * c2m_in_1).detach().cpu().numpy()

        B3 = (phimdx_b * n1 + phimdy_b * n2).detach().cpu().numpy()

        A_P = np.vstack((A3, p1 * B3, E1)) 
        b_P = np.vstack((fa3, p1 * fb3, fe1))
        wu_P = np.linalg.lstsq(A_P, b_P)[0] 
        phim.predict.weight.data = torch.from_numpy(wu_P[0:m]).T.to(device)

        # NS
        phimdx_1 = phimdx_in @ phim.predict.weight.data.T
        phimdy_1 = phimdy_in @ phim.predict.weight.data.T

        C1[:, 0:m] = (u1mdt_in + u1m_in_1 * u1mdx_in + u2m_in_1 * u1mdy_in - nv * (
                u1mdxx_in + u1mdyy_in)).detach().cpu().numpy()
        C1[:, m:2 * m] = (pmdx_in).detach().cpu().numpy()
        fc1 = (- phimdx_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()

        C2[:, 0:m] = (u2mdt_in + u1m_in_1 * u2mdx_in + u2m_in_1 * u2mdy_in - nv * (
                u2mdxx_in + u2mdyy_in)).detach().cpu().numpy()
        C2[:, m:2 * m] = (pmdy_in).detach().cpu().numpy()
        fc2 = (- phimdy_1 * (z1 * c1m_in_1 + z2 * c2m_in_1)).detach().cpu().numpy()

        B4[:, 0:m] = (u1m_b).detach().cpu().numpy()
        B5[:, 0:m] = (u2m_b).detach().cpu().numpy()
        I3[:, 0:m] = (u1m_i).detach().cpu().numpy()
        I4[:, 0:m] = (u2m_i).detach().cpu().numpy()

        A_NS = np.vstack((C1, C2, p1 * B4, p1 * B5, p1 * I3, p1 * I4, E2)) 
        b_NS = np.vstack((fc1, fc2, p1 * fb4, p1 * fb5, p1 * fi3, p1 * fi4, fe2))
        # b_NS = np.vstack((fc1, fc2, p1*fb4, p1*fb5, 10000*fb6, p1*fi3, p1*fi4))
        wu_NS = np.linalg.lstsq(A_NS, b_NS)[0] 
        um.predict.weight.data = torch.from_numpy(wu_NS[0:m]).T.to(device)
        pm.predict.weight.data = torch.from_numpy(wu_NS[m:2 * m]).T.to(device)

        end = time.perf_counter()
        total_time += (end - start)

        torch.save(c1m, legacy_output_path(f'PNPNS_code/model_ex2/c1m_{k+1}.pth'))
        torch.save(c2m, legacy_output_path(f'PNPNS_code/model_ex2/c2m_{k+1}.pth'))
        # torch.save(phim, legacy_output_path('PNPNS_code/model_ex2/phim_{k+1}.pth'))
        torch.save(um.state_dict(), legacy_output_path(f'PNPNS_code/model_ex2/um_weights_{k+1}.pth'))
        # torch.save(pm, legacy_output_path('PNPNS_code/model_ex2/pm_{k+1}.pth'))
        scaler['scaler_c1'][(k+1)*Nre:(k+2)*Nre] = s_c1
        scaler['scaler_c2'][(k+1)*Nre:(k+2)*Nre] = s_c2
        scaler['scaler_phi'][(k+1)*Nre:(k+2)*Nre] = s_phi

        for i in range(Nre):
            x2_plot = torch.cat((x2, xt_re[i] * torch.ones(x2.shape[0], 1)), dim=1)
            x2_plot.requires_grad_(True)
            c1m_plot = s_c1[i] * c1m(x2_plot)[1]
            c2m_plot = s_c2[i] * c2m(x2_plot)[1]
            u1m_plot = um(x2_plot)[3]
            u2m_plot = um(x2_plot)[4]
            results['c1_min'][i] = torch.min(c1m_plot)
            results['c2_min'][i] = torch.min(c2m_plot)
            results['m1'][i] = err_calculate.int_2d(s_c1[i]*c1m(x2_plot)[1], wr2, a, b, c, d)
            results['m2'][i] = err_calculate.int_2d(s_c2[i]*c2m(x2_plot)[1], wr2, a, b, c, d)
            results['Divergence_free'][i] = err_calculate.div_L2_norm_2d(u1m_plot, u2m_plot, x2_plot, wr2, a, b, c, d)
            phid1_plot = phim(x2_plot)[2]
            phidx_plot = ((phim.hidden.weight.data[:, 0:1].reshape([-1, m]) * phid1_plot)@phim.predict.weight.data.T)
            phidy_plot = ((phim.hidden.weight.data[:, 1:2].reshape([-1, m]) * phid1_plot)@phim.predict.weight.data.T)
            energy_terms = c1m_plot * torch.log(c1m_plot) + c2m_plot * torch.log(c2m_plot) + eps ** 2 / 2 * (phidx_plot ** 2 + phidy_plot ** 2) + 0.5*(u1m_plot**2 + u2m_plot**2)
            results['e'][i] = err_calculate.int_2d(energy_terms, wr2, a, b, c, d)

        PNPNS_plot_st_re_benchmark(xt_re, results, k+1)

        xt_re_all[(k + 1) * Nre:(k + 2) * Nre] = xt_re
        results_all['e_all'][(k + 1) * Nre:(k + 2) * Nre] = results['e']
        results_all['m1_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m1']
        results_all['m2_all'][(k + 1) * Nre:(k + 2) * Nre] = results['m2']
        results_all['c1_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c1_min']
        results_all['c2_min_all'][(k + 1) * Nre:(k + 2) * Nre] = results['c2_min']
        results_all['Divergence_free_all'][(k + 1) * Nre:(k + 2) * Nre] = results['Divergence_free']

        t1 = t2

    PNPNS_plot_st_re_all_benchmark(xt_re_all, results_all)

    print(total_time)

    return scaler
