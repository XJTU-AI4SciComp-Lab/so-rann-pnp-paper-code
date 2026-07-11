from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch

def c1(x):
    return torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1

def c2(x):
    return -torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1 

def phi(x):
    return torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])/math.pi**2

def u1(x):
    return math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 1:2])

def u2(x):
    return -math.pi*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2

def p(x):
    return torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])

def c1x(x):
    return -math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])

def c1y(x):
    return -math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])

def c2x(x):
    return math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])

def c2y(x):
    return math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])

def phix(x):
    return -torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])/math.pi

def phiy(x):
    return -torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])/math.pi

def u1x(x):
    return 2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])

def u1y(x):
    return 2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])**2*torch.cos(2*math.pi*x[:, 1:2])

def u2x(x):
    return -2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(2*math.pi*x[:, 0:1])

def u2y(x):
    return -2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 1:2])

def f1(x, z1, D1):
    return -D1*(z1*(-2*(torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1)*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.cos(math.pi*x[:, 1:2])**2 + torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])**2) - 2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])) - math.pi**2*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**3*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 1:2]) + math.pi**2*torch.sin(x[:, 2:3])**4*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**3*torch.cos(math.pi*x[:, 0:1]) + 2*torch.sin(x[:, 2:3])*torch.cos(x[:, 2:3])*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) 

def f2(x, z2, D2):
    return -D2*(z2*(-2*(-torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1)*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) - torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.cos(math.pi*x[:, 1:2])**2 - torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])**2) + 2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])) + math.pi**2*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**3*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 1:2]) - math.pi**2*torch.sin(x[:, 2:3])**4*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**3*torch.cos(math.pi*x[:, 0:1]) - 2*torch.sin(x[:, 2:3])*torch.cos(x[:, 2:3])*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) 

def f3(x, eps, z1, z2):
    return 2*eps**2*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) - z1*(torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1) - z2*(-torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1)

def g1(x, z1, z2, nu):
    return -nu*(-2*math.pi**3*(torch.sin(math.pi*x[:, 0:1])**2 - torch.cos(math.pi*x[:, 0:1])**2)*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 1:2]) - 4*math.pi**3*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 1:2])) - (z1*(torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1) + z2*(-torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1))*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])/math.pi + 2*math.pi**3*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**3*torch.sin(2*math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1]) - 2*math.pi**3*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(2*math.pi*x[:, 1:2]) + math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1]) + 2*math.pi*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(x[:, 2:3]) 

def g2(x, z1, z2, nu):
    return -nu*(2*math.pi**3*(torch.sin(math.pi*x[:, 1:2])**2 - torch.cos(math.pi*x[:, 1:2])**2)*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 0:1]) + 4*math.pi**3*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2) - (z1*(torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1) + z2*(-torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1))*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])/math.pi - 2*math.pi**3*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(2*math.pi*x[:, 0:1]) + 2*math.pi**3*torch.sin(x[:, 2:3])**4*torch.sin(2*math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])**3*torch.cos(math.pi*x[:, 1:2]) + math.pi*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) - 2*math.pi*torch.sin(x[:, 2:3])*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(x[:, 2:3]) 

def g3(x, z1, z2, nu):
    return 16*math.pi**4*nu*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1]) - 16*math.pi**4*nu*torch.sin(x[:, 2:3])**2*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 1:2]) - 2*(z1*(torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1) + z2*(-torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 1.1))*torch.sin(x[:, 2:3])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) - (-math.pi*z1*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + math.pi*z2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]))*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2])/math.pi - (-math.pi*z1*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1]) + math.pi*z2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1]))*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 1:2])*torch.cos(math.pi*x[:, 0:1])/math.pi - 2*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**4*torch.sin(2*math.pi*x[:, 1:2])**2 - 8*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(2*math.pi*x[:, 0:1])*torch.cos(2*math.pi*x[:, 1:2]) - 4*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(2*math.pi*x[:, 0:1])*torch.cos(math.pi*x[:, 1:2]) + 6*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])**2*torch.sin(2*math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])**2 - 4*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(math.pi*x[:, 0:1])*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 0:1])*torch.cos(2*math.pi*x[:, 1:2]) - 2*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(2*math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])**4 + 6*math.pi**4*torch.sin(x[:, 2:3])**4*torch.sin(2*math.pi*x[:, 0:1])**2*torch.sin(math.pi*x[:, 1:2])**2*torch.cos(math.pi*x[:, 1:2])**2 - 2*math.pi**2*torch.sin(x[:, 2:3])**2*torch.sin(math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2]) + 4*math.pi**2*torch.sin(x[:, 2:3])*torch.sin(math.pi*x[:, 0:1])*torch.sin(2*math.pi*x[:, 1:2])*torch.cos(x[:, 2:3])*torch.cos(math.pi*x[:, 0:1]) - 4*math.pi**2*torch.sin(x[:, 2:3])*torch.sin(2*math.pi*x[:, 0:1])*torch.sin(math.pi*x[:, 1:2])*torch.cos(x[:, 2:3])*torch.cos(math.pi*x[:, 1:2]) 

