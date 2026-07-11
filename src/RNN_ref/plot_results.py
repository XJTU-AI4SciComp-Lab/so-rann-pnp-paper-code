"""Reusable plotting helpers for PNP benchmark diagnostics."""

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from so_rann.paths import legacy_output_path
def plot_2d_scatter_1(X1):
    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], s=10, color='blue')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

def plot_2d_scatter_2(X1, X2):
    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], s=10, color='blue')
    plt.scatter(X2[:, 0], X2[:, 1], s=10, color='red')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

def plot_2d_line_2(X1, X2):
    iters = np.arange(1, len(X1) + 1)
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(iters, X1, marker='o', linewidth=1.5, label='vf  rL$^2$')
    plt.plot(iters, X2, marker='s', linewidth=1.5, label='us  rL$^2$')
    plt.yscale('log') 
    plt.xlabel('Outer iteration (ite1)')
    plt.ylabel('Relative $L^2$ error')
    plt.title('Convergence per ite1')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_curve_1d(a, b, step, f):
    x = np.arange(a, b, step)
    # y = np.sin(math.pi*np.sin(x))
    y = f(x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y)
    plt.show()

def plot_3d_scaler(a, b, c, d, e, f, fun):
    x = np.linspace(a, b, 50)
    y = np.linspace(c, d, 50)
    z = np.linspace(e, f, 50)
    X, Y = np.meshgrid(x, y)
    Z, Y2 = np.meshgrid(z, y)

    Z_x0 = fun(0, Y, Z)     
    Z_x1 = fun(1, Y, Z)     
    Z_y0 = fun(X, 0, Z)     
    Z_y1 = fun(X, 0.3, Z)   
    Z_z0 = fun(X, Y, 0)     
    Z_z1 = fun(X, Y, 0.1)   

    print(f"f(0, 0, 0) = {f(0, 0, 0)}")
    print(f"f(1, 0.3, 0.1) = {f(1, 0.3, 0.1)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf1 = ax.plot_surface(np.zeros_like(Y), Y, Z, facecolors=plt.cm.viridis(Z_x0), rstride=1, cstride=1, alpha=0.7, shade=False)

    surf2 = ax.plot_surface(np.ones_like(Y), Y, Z, facecolors=plt.cm.viridis(Z_x1), rstride=1, cstride=1, alpha=0.7, shade=False)

    surf3 = ax.plot_surface(X, np.zeros_like(Z), Z, facecolors=plt.cm.viridis(Z_y0), rstride=1, cstride=1, alpha=0.7, shade=False)

    surf4 = ax.plot_surface(X, np.ones_like(Z) * 0.3, Z, facecolors=plt.cm.viridis(Z_y1), rstride=1, cstride=1, alpha=0.7, shade=False)

    surf5 = ax.plot_surface(X, Y, np.zeros_like(X), facecolors=plt.cm.viridis(Z_z0), rstride=1, cstride=1, alpha=0.7, shade=False)

    surf6 = ax.plot_surface(X, Y, np.ones_like(X) * 0.1, facecolors=plt.cm.viridis(Z_z1), rstride=1, cstride=1, alpha=0.7, shade=False)

    ax.plot_wireframe(np.zeros_like(Y), Y, Z, color='k', linewidth=1)
    ax.plot_wireframe(np.zeros_like(Y), Y, np.ones_like(Z)*0.1, color='k', linewidth=1)

    ax.plot_wireframe(np.ones_like(Y), Y, Z, color='k', linewidth=1)
    ax.plot_wireframe(np.ones_like(Y), Y, np.ones_like(Z)*0.1, color='k', linewidth=1)

    ax.plot_wireframe(X, np.zeros_like(Z), Z, color='k', linewidth=1)
    ax.plot_wireframe(X, np.zeros_like(Z), np.ones_like(Z)*0.1, color='k', linewidth=1)

    ax.plot_wireframe(X, np.ones_like(Z)*0.3, Z, color='k', linewidth=1)
    ax.plot_wireframe(X, np.ones_like(Z)*0.3, np.ones_like(Z)*0.1, color='k', linewidth=1)

    ax.plot_wireframe(X, Y, np.zeros_like(X), color='k', linewidth=1)
    ax.plot_wireframe(X, Y, np.ones_like(X)*0.1, color='k', linewidth=1)

    ax.plot_wireframe(X, Y, np.ones_like(X)*0.1, color='k', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title("""Optimized plotting helper.""")

    ax.view_init(30, 30)

    ax.set_box_aspect([1, 0.3, 0.1]) 

    norm = Normalize(vmin=0, vmax=0.00001) 
    mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([]) 
    fig.colorbar(mappable, ax=ax, shrink=0.8)

    plt.show()





def PNP_plot_st_t_benchmark(xt_re, results, k, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min', 'c1_min'),
        (ax2, 'c2_min', 'c2_min'),
        (ax3, 'm1', 'mass1'),
        (ax4, 'm2', 'mass2'),
        (ax5, 'e_bar', 'energy_bar'),
        (ax6, 'e', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-t-ex{j}_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_t_all_benchmark(xt_re, results, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min_all', 'c1_min'),
        (ax2, 'c2_min_all', 'c2_min'),
        (ax3, 'm1_all', 'mass1'),
        (ax4, 'm2_all', 'mass2'),
        (ax5, 'e_bar_all', 'energy_bar'),
        (ax6, 'e_all', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-t-ex{j}-all.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_benchmark(xt_re, results, k, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min', 'c1_min'),
        (ax2, 'c2_min', 'c2_min'),
        (ax3, 'm1', 'mass1'),
        (ax4, 'm2', 'mass2'),
        (ax5, 'e_bar', 'energy_bar'),
        (ax6, 'e', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-re-ex{j}_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_all_benchmark(xt_re, results, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min_all', 'c1_min'),
        (ax2, 'c2_min_all', 'c2_min'),
        (ax3, 'm1_all', 'mass1'),
        (ax4, 'm2_all', 'mass2'),
        (ax5, 'e_bar_all', 'energy_bar'),
        (ax6, 'e_all', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-re-ex{j}-all.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_benchmark_p(xt_re, results, k, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min', 'c1_min'),
        (ax2, 'c2_min', 'c2_min'),
        (ax3, 'm1', 'mass1'),
        (ax4, 'm2', 'mass2'),
        (ax5, 'e_bar', 'energy_bar'),
        (ax6, 'e', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-re-p-ex{j}_{k}.eps'), bbox_inches='tight')
    plt.close(fig)

def PNP_plot_st_re_all_benchmark_p(xt_re, results, j):
    """Optimized plotting helper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "font.family": "Times New Roman",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    plots_config = [
        (ax1, 'c1_min_all', 'c1_min'),
        (ax2, 'c2_min_all', 'c2_min'),
        (ax3, 'm1_all', 'mass1'),
        (ax4, 'm2_all', 'mass2'),
        (ax5, 'e_bar_all', 'energy_bar'),
        (ax6, 'e_all', 'energy_pred'),
    ]

    for ax, data_key, title in plots_config:
        ax.plot(xt_re, results[data_key])
        ax.set(xlabel='t', ylabel='L2_error' if 'error' in data_key else 'value', title=title)
        ax.grid(True, alpha=0.3)

    fig.savefig(legacy_output_path(f'PNP_code/PNP_results_ex{j}/PNP-st-re-p-ex{j}-all.eps'), bbox_inches='tight')
    plt.close(fig)
