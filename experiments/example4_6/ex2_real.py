from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch

def c1(x):
    return torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1

def c2(x):
    return -torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1

def u1(x):
    return math.pi*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 1:2])

def u2(x):
    return -math.pi*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2
