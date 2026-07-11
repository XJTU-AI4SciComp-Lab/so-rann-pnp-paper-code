from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch

def c1(x):
    return 3*x[:, 0:1]**2 - 2*x[:, 0:1]**3 + 3*x[:, 1:2]**2 - 2*x[:, 1:2]**3

def c2(x):
    return 3*(3*x[:, 0:1]**2 - 2*x[:, 0:1]**3)*(3*x[:, 1:2]**2 - 2*x[:, 1:2]**3) + 9/4

def c3(x):
    return x[:, 0:1]**2*(1 - x[:, 0:1])**2 + x[:, 1:2]**2*(1 - x[:, 1:2])**2 + 14/15

# def c1x(x):
#     return 6*x[:, 0:1]*(1 - x[:, 0:1])
#
# def c1y(x):
#     return 6*x[:, 1:2]*(1 - x[:, 1:2])
#
# def c2x(x):
#     return 18*x[:, 0:1]*(1 - x[:, 0:1])*(3*x[:, 1:2]**2 - 2*x[:, 1:2]**3)
#
# def c2y(x):
#     return 18*x[:, 1:2]*(1 - x[:, 1:2])*(3*x[:, 0:1]**2 - 2*x[:, 0:1]**3)
#
# def c3x(x):
#     return 2*x[:, 0:1]*(1 - x[:, 0:1])*(1 - 2*x[:, 0:1])
#
# def c3y(x):
#     return 6*x[:, 1:2]*(1 - x[:, 1:2])*(1 - 2*x[:, 1:2])
