from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from so_rann.paths import legacy_output_path, result_path

# EX3（SIAM JNA ex.5.2）（benchmark test）

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc
from RNN_ref import RNN_funs
from RNN_ref import sampling_points
import torch
import numpy as np
import argparse
import PNP_assemble_st_re_ex3

torch.manual_seed(42)
np.random.seed(42)


def evaluate_on_grid(model, points, scale=1.0, batch_size=1000):
    """Evaluate the scalar network output in batches to keep peak memory low."""
    model.eval()
    values = []
    scale_value = torch.as_tensor(scale, dtype=points.dtype, device=points.device)
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            batch = points[start:start + batch_size]
            values.append((scale_value * model(batch)[1]).detach().cpu())
    return torch.cat(values, dim=0).numpy()


def add_field(ax, values, a, b, c, d, cmap):
    """Draw one 200 x 200 field and attach its colorbar."""
    image = values.reshape(200, 200)
    h = ax.imshow(
        image,
        interpolation='nearest',
        cmap=cmap,
        extent=[a, b, c, d],
        origin='lower',
        aspect=1,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    return h


def get_args(): 
    parser = argparse.ArgumentParser(description='Train the pde net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r_c1', '--range1', default=2, help='Range of initial weights', type=float)
    parser.add_argument('-r_c2', '--range2', default=2, help='Range of initial weights', type=float)
    parser.add_argument('-r_phi', '--range3', default=2, help='Range of initial weights', type=float)
    parser.add_argument('-n', '--numbers1', default=20, help='points on unit edge', type=int)
    parser.add_argument('-Nt', '--numbers_t', default=10, help='points of time', type=int)
    parser.add_argument('-Nre', '--numbers_re', default=51, help='points of time', type=int)
    parser.add_argument('-n_int', '--numbers2', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=800, help='number of hidden layer nodes', type=int)
    parser.add_argument('-tt', '--times', default=30, help='times of Picard iteration', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('--plot_only', action='store_true', help='only regenerate Figure 10 from saved checkpoints and scaler')
    parser.add_argument('--plot_batch_size', default=1000, help='batch size used when evaluating Figure 10 fields', type=int)
    return parser.parse_args()


def main(r_c1, r_c2, r_phi, n, Nt, Nre, n_int, m, tt, p1, device, plot_only=False, plot_batch_size=1000): 
    a = -2
    b = 2
    c = -2
    d = 2
    t1 = 0
    t2 = 1
    threshold = 1e-6
    D1 = 1
    D2 = 1
    z1 = 1
    z2 = -1
    eps = 1

    xr2 = sampling_points.get_GL_points_2d(n_int)
    wr2 = sampling_points.get_weights_2d(n_int)
    wr2 = wr2.reshape([n_int ** 2, 1])
    wr2 = wr2.to(device)
    xr2 = xr2.to(device)
    x2 = torch.zeros_like(xr2)
    x2[:, 0:1] = (b - a) / 2 * xr2[:, 0:1] + (a + b) / 2
    x2[:, 1:2] = (d - c) / 2 * xr2[:, 1:2] + (c + d) / 2

    s = (t2 - t1) / Nt
    scaler_path = result_path('PNP_code', 'example4_3', 'scaler.pt')

    if plot_only:
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Cannot run --plot_only because {scaler_path} does not exist. "
                "Run the full script once with the updated code to save the scaler."
            )
        print(f"Loading saved scaler from {scaler_path}", flush=True)
        scaler = torch.load(scaler_path, map_location=device, weights_only=False)
    else:
        print(
            f"Starting Example 4.3 solve with n={n}, Nt={Nt}, Nre={Nre}, "
            f"n_int={n_int}, m={m}, tt={tt}",
            flush=True,
        )
        # RaNN
        # c1m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
        # c1m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
        # SO-RaNN
        c1m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
        RNN_funs.weights_init_uniform_0(c1m, -r_c1, r_c1)
        # RaNN
        # c2m = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
        # SO-RaNN
        # c2m = RNN_funs.rnn_gauss_positive_exact_1d(3, m, 1).to(device)
        c2m = RNN_funs.rnn_gauss_positive1_exact_1d(3, m, 1).to(device)
        RNN_funs.weights_init_uniform_0(c2m, -r_c2, r_c2)
        # phim = RNN_funs.rnn_sin_exact_1d(3, m, 1).to(device)

        phim = RNN_funs.rnn_gauss_exact_1d(3, m, 1).to(device)
        RNN_funs.weights_init_uniform_0(phim, -r_phi, r_phi)

        # SO-RaNN
        scaler = PNP_assemble_st_re_ex3.pnp_ex3_picard_st_G_remass_threshold_plus(n, n_int, Nt, Nre, s, x2, wr2, m, c1m, c2m, phim, a, b, c, d, t1, D1, D2, z1, z2, eps, threshold, tt, p1)
        torch.save(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}", flush=True)

    c1m_0 = torch.load(legacy_output_path('PNP_code/model_ex3/c1m_0.pth'), map_location=device, weights_only=False)
    c2m_0 = torch.load(legacy_output_path('PNP_code/model_ex3/c2m_0.pth'), map_location=device, weights_only=False)
    phim_0 = torch.load(legacy_output_path('PNP_code/model_ex3/phim_0.pth'), map_location=device, weights_only=False)

    c1m_T = torch.load(legacy_output_path(f'PNP_code/model_ex3/c1m_{Nt-1}.pth'), map_location=device, weights_only=False)
    c2m_T = torch.load(legacy_output_path(f'PNP_code/model_ex3/c2m_{Nt-1}.pth'), map_location=device, weights_only=False)
    phim_T = torch.load(legacy_output_path(f'PNP_code/model_ex3/phim_{Nt-1}.pth'), map_location=device, weights_only=False)

    x1 = torch.linspace(a, b, 200)
    x2 = torch.linspace(c, d, 200)
    Y, X = torch.meshgrid(x2, x1, indexing='ij') 
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    Z0 = torch.cat((Z, torch.zeros(40000, 1)), dim=1).to(device)
    Z1 = torch.cat((Z, 0.04*torch.ones(40000, 1)), dim=1).to(device)
    Z2 = torch.cat((Z, 0.2*torch.ones(40000, 1)), dim=1).to(device)
    Z3 = torch.cat((Z, 1*torch.ones(40000, 1)), dim=1).to(device)

    print("Preparing Figure 10 fields...", flush=True)
    pred1 = evaluate_on_grid(c1m_0, Z0, scaler['scaler_c1'][0], plot_batch_size)
    pred5 = evaluate_on_grid(c2m_0, Z0, scaler['scaler_c2'][0], plot_batch_size)
    pred9 = evaluate_on_grid(phim_0, Z0, batch_size=plot_batch_size)

    pred4 = evaluate_on_grid(c1m_T, Z3, scaler['scaler_c1'][-1], plot_batch_size)
    pred8 = evaluate_on_grid(c2m_T, Z3, scaler['scaler_c2'][-1], plot_batch_size)
    pred12 = evaluate_on_grid(phim_T, Z3, batch_size=plot_batch_size)
    del c1m_0, c2m_0, phim_0, c1m_T, c2m_T, phim_T
    gc.collect()

    t0 = t1
    for i in range(Nt):
        if t0 < 0.04 <= t0 + s:
            c1m = torch.load(legacy_output_path(f'PNP_code/model_ex3/c1m_{i}.pth'), map_location=device, weights_only=False)
            c2m = torch.load(legacy_output_path(f'PNP_code/model_ex3/c2m_{i}.pth'), map_location=device, weights_only=False)
            phim = torch.load(legacy_output_path(f'PNP_code/model_ex3/phim_{i}.pth'), map_location=device, weights_only=False)

            index1 = int((0.04 - t0)/s*Nre)
            pred2 = evaluate_on_grid(c1m, Z1, scaler['scaler_c1'][i*Nre + index1], plot_batch_size)
            pred6 = evaluate_on_grid(c2m, Z1, scaler['scaler_c2'][i*Nre + index1], plot_batch_size)
            pred10 = evaluate_on_grid(phim, Z1, batch_size=plot_batch_size)
            del c1m, c2m, phim
            gc.collect()

            print(i)
        if t0 - 1e-3 < 0.2 <= t0 + s - 1e-3:
            c1m = torch.load(legacy_output_path(f'PNP_code/model_ex3/c1m_{i}.pth'), map_location=device, weights_only=False)
            c2m = torch.load(legacy_output_path(f'PNP_code/model_ex3/c2m_{i}.pth'), map_location=device, weights_only=False)
            phim = torch.load(legacy_output_path(f'PNP_code/model_ex3/phim_{i}.pth'), map_location=device, weights_only=False)

            index2 = int((0.2 - t0)/s*Nre)
            pred3 = evaluate_on_grid(c1m, Z2, scaler['scaler_c1'][i*Nre + index2], plot_batch_size)
            pred7 = evaluate_on_grid(c2m, Z2, scaler['scaler_c2'][i*Nre + index2], plot_batch_size)
            pred11 = evaluate_on_grid(phim, Z2, batch_size=plot_batch_size)
            del c1m, c2m, phim
            gc.collect()

            print(i)
        t0 = t0 + s

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='w')
    for ax, values in zip(
        axes.flat,
        [pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12],
    ):
        add_field(ax, values, a, b, c, d, cmap)
    plt.rcParams['font.sans-serif'] = ['Times New Roman'] 
    plt.rcParams['axes.unicode_minus'] = False 
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    figure10_path = legacy_output_path('PNP_code/PNP_ex3_paper/PNP_r_ex3_paper/SO-RaNN/PNP-st-re-ex3-p.eps')
    png_path = figure10_path.with_suffix('.png')
    print(f"Saving Figure 10 PNG to {png_path}", flush=True)
    fig.savefig(png_path, dpi=300)
    print(f"Saving Figure 10 EPS to {figure10_path}", flush=True)
    fig.savefig(figure10_path)
    plt.close(fig)

if __name__ == '__main__':
    args = get_args() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    main(r_c1=args.range1, r_c2=args.range2, r_phi=args.range3, n=args.numbers1, Nt=args.numbers_t, Nre=args.numbers_re,
         n_int=args.numbers2, m=args.neurons, tt=args.times, p1=args.penalty1, device=device,
         plot_only=args.plot_only, plot_batch_size=args.plot_batch_size)
