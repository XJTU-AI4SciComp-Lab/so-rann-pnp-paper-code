"""Finite-difference derivative approximations used by experiments."""

import torch

def fdpx(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,0:1]=1
    ux = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return ux

def fdpy(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,1:2]=1
    uy = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uy

def fdpt(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2:3]=1
    ut = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return ut

def fdpx_1(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,0:1]=1
    ux = (model(xr+h*ei)[1] - model(xr-h*ei)[1])/(2*h)
    return ux

def fdpy_1(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,1:2]=1
    uy = (model(xr+h*ei)[1] - model(xr-h*ei)[1])/(2*h)
    return uy

def fdpt_1(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2:3]=1
    ut = (model(xr+h*ei)[1] - model(xr-h*ei)[1])/(2*h)
    return ut

def fdpxx(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,0:1]=1
    uxx = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return uxx

def fdpyy(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,1:2]=1
    uyy = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return uyy

def fdptt(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2:3]=1
    utt = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return utt

def fdpxy(model,xr,h):
    ei11 = -1*torch.ones_like(xr)
    ei12 = torch.zeros_like(xr)
    ei21 = torch.zeros_like(xr)
    ei22 = torch.ones_like(xr)
    ei11[:, 2:3] = 0
    ei22[:, 2:3] = 0
    ei12[:, 1:2] = -1
    ei12[:, 0:1] = 1
    ei21[:, 1:2] = 1
    ei21[:, 0:1] = -1
    u1xy = (model(xr+h*ei22)[0] + model(xr+h*ei11)[0] - model(xr+h*ei12)[0] - model(xr+h*ei21)[0])/(4*h**2)
    return u1xy

def fdptx(model,xr,h):
    ei11 = -1*torch.ones_like(xr)
    ei12 = torch.zeros_like(xr)
    ei21 = torch.zeros_like(xr)
    ei22 = torch.ones_like(xr)
    ei11[:, 1:2] = 0
    ei22[:, 1:2] = 0
    ei12[:, 2:3] = -1
    ei12[:, 0:1] = 1
    ei21[:, 2:3] = 1
    ei21[:, 0:1] = -1
    u1tx = (model(xr+h*ei22)[0] + model(xr+h*ei11)[0] - model(xr+h*ei12)[0] - model(xr+h*ei21)[0])/(4*h**2)
    return u1tx

def fdpty(model,xr,h):
    ei11 = -1*torch.ones_like(xr)
    ei12 = torch.zeros_like(xr)
    ei21 = torch.zeros_like(xr)
    ei22 = torch.ones_like(xr)
    ei11[:, 0:1] = 0
    ei22[:, 0:1] = 0
    ei12[:, 2:3] = -1
    ei12[:, 1:2] = 1
    ei21[:, 2:3] = 1
    ei21[:, 1:2] = -1
    u2tx = (model(xr+h*ei22)[0] + model(xr+h*ei11)[0] - model(xr+h*ei12)[0] - model(xr+h*ei21)[0])/(4*h**2)
    return u2tx

def fdpxx_1(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,0:1]=1
    uxx = (model(xr+h*ei)[1] + model(xr-h*ei)[1] - 2*model(xr)[1])/(h**2)
    return uxx

def fdpyy_1(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,1:2]=1
    uyy = (model(xr+h*ei)[1] + model(xr-h*ei)[1] - 2*model(xr)[1])/(h**2)
    return uyy
