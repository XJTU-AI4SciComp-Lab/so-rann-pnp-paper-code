from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch

def c1(x):
    return torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1

def c2(x):
    return -torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1

def phi(x):
    return torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])/math.pi**2

def c1x(x):
    return math.pi*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])

def c1y(x):
    return math.pi*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])

def c2x(x):
    return -math.pi*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])

def c2y(x):
    return -math.pi*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])

def phix(x):
    return torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])/math.pi

def phiy(x):
    return torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])/math.pi

def f1(x, D1, z1):
    return -D1*z1*(-2*(torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1)*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])**2*torch.cos(math.pi*x[:, 1:2])**2 + torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])**2) + 2*math.pi**2*D1*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])*torch.cos(x[:, 2:3])

def f2(x, D2, z2):
    return -D2*z2*(-2*(-torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1)*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) - torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])**2*torch.cos(math.pi*x[:, 1:2])**2 - torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])**2) - 2*math.pi**2*D2*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) - torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])*torch.cos(x[:, 2:3])

def f3(x, eps, z1, z2):
    return 2*eps**2*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) - z1*(torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1) - z2*(-torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 1.1)
