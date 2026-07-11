"""Quadrature-based norms and error metrics."""

import torch
from torch import autograd

torch.set_default_dtype(torch.float64)

def int_2d(exact, wr, a, b, c, d):
    exact_L2 = (b - a) * (d - c) / 4 * torch.sum(exact * wr)
    return exact_L2

def int_3d(exact, wr, a, b, c, d, e, f):
    exact_L2 = (b - a) * (d - c) * (f - e)/ 8 * torch.sum(exact * wr)
    return exact_L2

def L2_norm_2d(exact, wr, a, b, c, d):
    exact_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(exact, 2) * wr))
    return exact_L2

def L2_norm_3d(exact, wr, a, b, c, d, e, f):
    exact_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum(torch.pow(exact, 2) * wr))
    return exact_L2

def L2_norm_3d_vec2(exact1, exact2, wr, a, b, c, d, e, f):
    exact_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum((torch.pow(exact1, 2) + torch.pow(exact2, 2)) * wr))
    return exact_L2

def div_L2_norm_2d(output1, output2, x, wr, a, b, c, d):
    grad_output1 = autograd.grad(outputs=output1, inputs=x, grad_outputs=torch.ones_like(output1), create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_output2 = autograd.grad(outputs=output2, inputs=x, grad_outputs=torch.ones_like(output2), create_graph=True, retain_graph=True, only_inputs=True)[0]
    output1dx = grad_output1[:, 0:1]
    output2dy = grad_output2[:, 1:2]
    divergence_L2_norm = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output1dx + output2dy, 2) * wr))
    return divergence_L2_norm

def curl_L2_norm_2d(output1, output2, x, wr, a, b, c, d):
    grad_output1 = autograd.grad(outputs=output1, inputs=x, grad_outputs=torch.ones_like(output1), retain_graph=True, only_inputs=True)[0]
    grad_output2 = autograd.grad(outputs=output2, inputs=x, grad_outputs=torch.ones_like(output2), retain_graph=True, only_inputs=True)[0]
    output1dy = grad_output1[:, 1:2]
    output2dx = grad_output2[:, 0:1]
    output_curl_L2_norm = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output2dx - output1dy, 2) * wr))
    return output_curl_L2_norm

def err_L2_2d(output, exact, wr, a, b, c, d):
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output - exact, 2) * wr))
    return err_L2

def err_L2_3d(output, exact, wr, a, b, c, d, e, f):
    err_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum(torch.pow(output - exact, 2) * wr))
    return err_L2

def err_L2_2d_vec2(output1, exact1, output2, exact2, wr, a, b, c, d):
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum((torch.pow(output1 - exact1, 2) + torch.pow(output2 - exact2, 2)) * wr))
    return err_L2

def err_L2_3d_vec2(output1, exact1, output2, exact2, wr, a, b, c, d, e, f):
    err_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum((torch.pow(output1 - exact1, 2) + torch.pow(output2 - exact2, 2)) * wr))
    return err_L2

def err_rH1_2d(output1, output2, exact1dx, exact1dy, exact2dx, exact2dy, x, wr, a, b, c, d):
    grad_output1 = autograd.grad(outputs=output1, inputs=x,
                            grad_outputs=torch.ones_like(output1),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_output2 = autograd.grad(outputs=output2, inputs=x,
                            grad_outputs=torch.ones_like(output2),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    output1dx = grad_output1[:, 0:1]
    output1dy = grad_output1[:, 1:2]
    output2dx = grad_output2[:, 0:1]
    output2dy = grad_output2[:, 1:2]
    err_H1 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(((torch.pow(output1dx - exact1dx, 2) + torch.pow(output1dy - exact1dy, 2) + torch.pow(output2dx - exact2dx, 2) + torch.pow(output2dy - exact2dy, 2)) * wr)))
    H1_norm = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(((torch.pow(exact1dx, 2) + torch.pow(exact1dy, 2) + torch.pow(exact2dx, 2) + torch.pow(exact2dy, 2)) * wr)))
    err_rH1 = err_H1 / H1_norm
    return err_rH1
