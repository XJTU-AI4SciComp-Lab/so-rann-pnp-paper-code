from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def c1(x):
    # r = torch.zeros(x.shape[0], 1)
    r = 1e-10*torch.ones(x.shape[0], 1)
    mask1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2 < 0.25
    mask2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2 < 0.25
    r[mask1] = 1
    r[mask2] = 0.5
    return r

def c2(x):
    # r = torch.zeros(x.shape[0], 1)
    r = 1e-10*torch.ones(x.shape[0], 1)
    mask1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2 < 0.25
    mask2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2 < 0.25
    r[mask1] = 0.5
    r[mask2] = 1
    return r

def c1x(x):
    return torch.zeros(x.shape[0], 1)

def c1y(x):
    return torch.zeros(x.shape[0], 1)

def c2x(x):
    return torch.zeros(x.shape[0], 1)

def c2y(x):
    return torch.zeros(x.shape[0], 1)

# def c1(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     h1 = 0.5*(1 - torch.tanh((r1 - 0.25)/eps))
#     h2 = 0.5*(1 - torch.tanh((r2 - 0.25)/eps))
#     return h1 + 0.5*h2 + 1e-10
#
# def c2(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     h1 = 0.5*(1 - torch.tanh((r1 - 0.25)/eps))
#     h2 = 0.5*(1 - torch.tanh((r2 - 0.25)/eps))
#     return 0.5*h1 + h2 + 1e-10
#
# def c1x(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     return -1/eps*((x[:, 0:1] - 0.5)*(1 - torch.tanh((r1 - 0.25)/eps)**2) + 0.5*(x[:, 0:1] + 0.5)*(1 - torch.tanh((r2 - 0.25)/eps)**2))
#
# def c1y(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     return -1/eps*((x[:, 1:2] - 0.5)*(1 - torch.tanh((r1 - 0.25)/eps)**2) + 0.5*(x[:, 1:2] + 0.5)*(1 - torch.tanh((r2 - 0.25)/eps)**2))
#
# def c2x(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     return -1/eps*(0.5*(x[:, 0:1] - 0.5)*(1 - torch.tanh((r1 - 0.25)/eps)**2) + (x[:, 0:1] + 0.5)*(1 - torch.tanh((r2 - 0.25)/eps)**2))
#
# def c2y(x, eps = 1e-1):
#     r1 = (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2
#     r2 = (x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2
#     return -1/eps*(0.5*(x[:, 1:2] - 0.5)*(1 - torch.tanh((r1 - 0.25)/eps)**2) + (x[:, 1:2] + 0.5)*(1 - torch.tanh((r2 - 0.25)/eps)**2))

# x = np.linspace(-2, 2, 300)
# y = np.linspace(-2, 2, 300)
# X, Y = np.meshgrid(x, y)
#
# points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
#
# with torch.no_grad():
#     # Z1 = c1(points).numpy().reshape(X.shape)
#     # Z2 = c2(points).numpy().reshape(X.shape)
#
#     Z1 = c1(points, eps=5e-3).numpy().reshape(X.shape)
#     Z2 = c2(points, eps=5e-3).numpy().reshape(X.shape)
#
#     # Z1 = c1x(points, eps=5e-3).numpy().reshape(X.shape)
#     # Z2 = c1y(points, eps=5e-3).numpy().reshape(X.shape)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#
# im1 = axes[0].imshow(Z1, extent=[-2, 2, -2, 2], origin='lower', cmap='RdYlBu_r')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# axes[0].set_title('c1: h1 + 0.5·h2')
# plt.colorbar(im1, ax=axes[0])
#
# im2 = axes[1].imshow(Z2, extent=[-2, 2, -2, 2], origin='lower', cmap='RdYlBu_r')
# axes[1].set_xlabel('x')
# axes[1].set_ylabel('y')
# axes[1].set_title('c2: 0.5·h1 + h2')
# plt.colorbar(im2, ax=axes[1])
#
#
# plt.tight_layout()
# plt.show()
