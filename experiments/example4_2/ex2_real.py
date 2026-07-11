from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch

def c1(x):
    return -4*(x[:, 0:1]**2*(1 - x[:, 0:1])**2 + x[:, 1:2]**2*(1 - x[:, 1:2])**2) + 19/15

def c2(x):
    return -15*(x[:, 0:1]**4*(1 - x[:, 0:1])**4 + x[:, 1:2]**4*(1 - x[:, 1:2])**4) + 22/21

def c1x(x):
    return -4*(2*x[:, 0:1]*(1 - x[:, 0:1])*(1 - 2*x[:, 0:1]))

def c1y(x):
    return -4*(2*x[:, 1:2]*(1 - x[:, 1:2])*(1 - 2*x[:, 1:2]))

def c2x(x):
    return -15*(4*x[:, 0:1]**3*(1 - x[:, 0:1])**3*(1 - 2*x[:, 0:1]))

def c2y(x):
    return -15*(4*x[:, 1:2]**3*(1 - x[:, 1:2])**3*(1 - 2*x[:, 1:2]))
