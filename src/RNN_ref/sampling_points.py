"""Sampling utilities for Gauss-Legendre rules and rectangular domains."""

from numpy.polynomial.legendre import leggauss
from itertools import product
import torch
import math
# torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
def get_GL_points_2d(N):
    points, weights = leggauss(N) 
    l = []
    for item in product(points, points):
        l.append(list(item))
    xy = torch.tensor(l)
    return xy


def get_weights_2d(N):
    points, weights = leggauss(N) 
    l = []
    for item in product(weights, weights):
        l.append(list(item))
    xy = torch.tensor(l)
    gz = torch.mul(xy[:, 0], xy[:,1])
    return gz

def get_GL_points_3d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(points, points, points):
        l.append(list(item))
    xy = torch.tensor(l)
    return xy

def get_weights_3d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(weights, weights, weights):
        l.append(list(item))
    xy = torch.tensor(l)
    gz = torch.mul(torch.mul(xy[:, 0], xy[:,1]), xy[:,2])
    return gz

def get_GL_points_4d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(points, points, points, points):
        l.append(list(item))
    xy = torch.tensor(l)
    return xy

def get_weights_4d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(weights, weights, weights, weights):
        l.append(list(item))
    xy = torch.tensor(l)
    gz = torch.mul(torch.mul(torch.mul(xy[:, 0], xy[:,1]), xy[:,2]), xy[:,3])
    return gz
def get_points_1d_flower(N, c1, c2, omega):
    theta = 2*math.pi*torch.rand(N,1)
    r = 0.4 + 0.2*torch.sin(omega*theta)
    index1 = c1 + r*torch.cos(theta)
    index2 = c2 + r*torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    return xb, theta

def get_points_circle_t(N, c1, c2, r, t1, t2):
    theta = 2 * math.pi * torch.rand(N, 1)
    index1 = r * torch.cos(theta) + c1
    index2 = r * torch.sin(theta) + c2
    index3 = (t2 - t1) * torch.rand(N, 1) + t1
    xb = torch.cat((index1, index2, index3), dim=1)
    return xb

def get_points_circle(N, c1, c2, r):
    theta = 2 * math.pi * torch.rand(N, 1)
    index1 = r * torch.cos(theta) + c1
    index2 = r * torch.sin(theta) + c2
    xb = torch.cat((index1, index2), dim=1)
    return xb


def get_interface_points_circle_t(N, c1, c2, r, t1, t2):
    theta = 2 * math.pi * torch.rand(N, 1)
    index1 = r * torch.cos(theta) + c1
    index2 = r * torch.sin(theta) + c2
    index3 = (t2 - t1) * torch.rand(N, 1) + t1
    xb = torch.cat((index1, index2, index3), dim=1)
    n1 = torch.cos(theta)
    n2 = torch.sin(theta)
    return xb, n1, n2

def get_boundary_points_rec_2d(N1, N2, a, b, c, d):
    index1 = (b - a) * torch.rand(N1, 1) + a
    index2 = (d - c) * torch.rand(N2, 1) + c
    xb1 = torch.cat((index1, c * torch.ones_like(index1)), dim=1)
    xb2 = torch.cat((index1, d * torch.ones_like(index1)), dim=1)
    xb3 = torch.cat((a * torch.ones_like(index2), index2), dim=1)
    xb4 = torch.cat((b * torch.ones_like(index2), index2), dim=1)
    xb = torch.cat((xb1,xb2,xb3,xb4), dim=0)
    return xb
def get_interior_points_rec_2d(N, a, b, c, d):
    index1 = (b - a)*torch.rand(N, 1) + a
    index2 = (d - c)*torch.rand(N, 1) + c
    xb = torch.cat((index1, index2), dim=1)
    return xb

def get_boundary_points_rec_3d(N1, N2, N3, a, b, c, d, e, f):
    index1x = (b - a) * torch.rand(N1, 1) + a
    index1y = (d - c) * torch.rand(N1, 1) + c
    index2x = (b - a) * torch.rand(N2, 1) + a
    index2z = (f - e) * torch.rand(N2, 1) + e
    index3y = (d - c) * torch.rand(N3, 1) + c
    index3z = (f - e) * torch.rand(N3, 1) + e
    xb1 = torch.cat((index1x, index1y, e * torch.ones_like(index1x)), dim=1)
    xb2 = torch.cat((index1x, index1y, f * torch.ones_like(index1x)), dim=1)
    xb3 = torch.cat((index2x, c * torch.ones_like(index2x), index2z), dim=1)
    xb4 = torch.cat((index2x, d * torch.ones_like(index2x), index2z), dim=1)
    xb5 = torch.cat((a * torch.ones_like(index3y), index3y, index3z), dim=1)
    xb6 = torch.cat((b * torch.ones_like(index3y), index3y, index3z), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)
    return xb
def get_interior_points_rec_3d(N, a, b, c, d, e, f):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    index3 = (f-e)*torch.rand(N, 1) + e
    xb = torch.cat((index1, index2,index3), dim=1)
    return xb

def get_boundary_points_rec_2d_t(N, a, b, t1, t2):
    tt = (t2 - t1)*torch.rand(N,1) + t1
    xb1 = torch.cat((a * torch.ones(N,1), tt), dim=1)
    xb2 = torch.cat((b * torch.ones(N,1), tt), dim=1)
    xb = torch.cat((xb1, xb2), dim=0)
    return xb
def get_initial_points_rec_2d_t(N, a, b, t1):
    index1 = (b - a) * torch.rand(N, 1) + a
    xb = torch.cat((index1, t1*torch.ones((N,1))), dim=1)
    return xb


def get_boundary_points_rec_3d_t(N1, N2, a, b, c, d, t1, t2):
    index1 = (b - a) * torch.rand(N1, 1) + a
    index2 = (d - c) * torch.rand(N2, 1) + c
    index3 = (t2 - t1)*torch.rand(N1, 1) + t1
    index4 = (t2 - t1)*torch.rand(N2, 1) + t1
    xb1 = torch.cat((index1, c * torch.ones_like(index1), index3), dim=1)
    xb2 = torch.cat((index1, d * torch.ones_like(index1), index3), dim=1)
    xb3 = torch.cat((a * torch.ones_like(index2), index2, index4), dim=1)
    xb4 = torch.cat((b * torch.ones_like(index2), index2, index4), dim=1)
    xb = torch.cat((xb1,xb2,xb3,xb4), dim=0)
    return xb
def get_initial_points_rec_3d_t(N, a, b, c, d, t1):
    index1 = (b - a) * torch.rand(N, 1) + a
    index2 = (d - c) * torch.rand(N, 1) + c
    xi = torch.cat((index1, index2, t1*torch.ones((N,1))),dim=1)
    return xi


def get_boundary_points_rec_4d_t(N1, N2, N3, a, b, c, d, e, f, t1, t2):
    index12 = (d - c)*torch.rand(N1, 1) + c
    index13 = (f - e)*torch.rand(N1, 1) + e
    index21 = (b - a)*torch.rand(N2, 1) + a
    index23 = (f - e)*torch.rand(N2, 1) + e
    index31 = (b - a)*torch.rand(N3, 1) + a
    index32 = (d - c)*torch.rand(N3, 1) + c
    tt1 = (t2 - t1)*torch.rand(N1, 1) + t1
    tt2 = (t2 - t1)*torch.rand(N2, 1) + t1
    tt3 = (t2 - t1)*torch.rand(N3, 1) + t1
    xb1 = torch.cat((a * torch.ones(N1, 1), index12, index13, tt1), dim=1)
    xb2 = torch.cat((b * torch.ones(N1, 1), index12, index13, tt1), dim=1)
    xb3 = torch.cat((index21, c * torch.ones(N2, 1), index23, tt2), dim=1)
    xb4 = torch.cat((index21, d * torch.ones(N2, 1), index23, tt2), dim=1)
    xb5 = torch.cat((index31, index32, e * torch.ones(N3, 1), tt3), dim=1)
    xb6 = torch.cat((index31, index32, f * torch.ones(N3, 1), tt3), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)
    return xb
def get_initial_points_rec_4d_t(N, a, b, c, d, e, f, t1):
    index1 = (b - a) * torch.rand(N, 1) + a
    index2 = (d - c) * torch.rand(N, 1) + c
    index3 = (e - f) * torch.rand(N, 1) + e
    xb = torch.cat((index1, index2, index3, t1*torch.ones((N, 1))), dim=1)
    return xb
def get_interior_points_rec_4d_t(N, a, b, c, d, e, f, t1, t2):
    index1 = (b - a)*torch.rand(N, 1) + a
    index2 = (d - c)*torch.rand(N, 1) + c
    index3 = (f - e)*torch.rand(N, 1) + e
    index4 = (t2 - t1)*torch.rand(N, 1) + t1
    xb = torch.cat((index1, index2, index3, index4), dim=1)
    return xb

def get_points_uniform_2d(a, b, c, d, nx, ny):
    index1 = torch.linspace(a, b, nx)
    index2 = torch.linspace(c, d, ny)
    xb = torch.cartesian_prod(index1, index2)
    return xb

def get_points_uniform_3d(a, b, c, d, e, f, nx, ny, nz):
    index1 = torch.linspace(a+1e-10, b-1e-10, nx)
    index2 = torch.linspace(c+1e-10, d-1e-10, ny)
    index3 = torch.linspace(e+1e-10, f-1e-10, nz)
    xb = torch.cartesian_prod(index1, index2, index3)
    return xb

def get_points_uniform_4d(a, b, c, d, e, f, t1, t2, nx, ny, nz, nt):
    index1 = torch.linspace(a, b, nx)
    index2 = torch.linspace(c, d, ny)
    index3 = torch.linspace(e, f, nz)
    index4 = torch.linspace(t1, t2, nt)
    xb = torch.cartesian_prod(index1, index2, index3, index4)
    return xb
