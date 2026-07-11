"""Randomized neural network bases and helper functions."""

import torch
import torch.nn as nn
import math
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

def gaussian(x, mu=0.0, sigma=1.0):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

# input X: N*d1
# hidden layer W1: m*d1
# hidden layer bias: m   (input × weight^T + bias)
# predict layer W2: d2*m
# output Y: N*d2
# Y = X*(W1)^T*(W2)^T
class rnn_tanh_diff_1d_sca(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)            # output layer
        self.phi = torch.tanh                                                     # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                              # output of hidden layer
        y = self.predict(x)                                                       # linear output
        return x, y

class rnn_tanh_diff_2d_sca(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = torch.tanh

    def forward(self, x):
        x = self.phi(self.hidden(x))  # activation function for hidden layer
        y = self.predict(x)  # linear output
        y1 = y[:,0:1]
        y2 = y[:,1:2]
        return x,y,y1,y2

class rnn_sin_diff_2d_sca(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = torch.sin

    def forward(self, x):
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        return x, y

class rnn_guass_diff_2d_sca(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = gaussian

    def forward(self, x):
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        return x, y

class rnn_tanh_diff_2d_vec(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 1
        self.hidden2 = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 2
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output 1
        self.phi1 = torch.tanh
        self.phi2 = torch.tanh

    def forward(self, x):
        x1 = self.phi1(x)
        x2 = self.phi2(x)
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        y = torch.cat((y1,y2),dim = 1)
        return y, x1, x2, y1, y2


class rnn_tanh_diff_3d_vec(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 1
        self.hidden2 = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 2
        self.hidden3 = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 3
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output 1
        self.phi1 = torch.tanh
        self.phi2 = torch.tanh
        self.phi3 = torch.tanh

    def forward(self, x):
        x1 = self.phi1(x)
        x2 = self.phi2(x)
        x3 = self.phi3(x)
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        y3 = self.predict(x3)
        y = torch.cat((y1,y2,y3),dim = 1)
        return y, x1, x2, x3, y1, y2, y3


def int_1d_gaussian(x):
    return math.sqrt(math.pi / 2) * torch.erf(x / math.sqrt(2))

def int_2d_gaussian(x):
    erf_term = math.sqrt(math.pi / 2) * x * torch.erf(x / math.sqrt(2))
    exp_term = 0.5 * torch.exp(-0.5 * x ** 2)
    return erf_term + exp_term

def relu_like(x, eps: float = 1e-10) -> torch.Tensor:
    return torch.where(x < eps, torch.tensor(eps, dtype=x.dtype, device=x.device), x)

class rnn_tanh_exact_1d(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)            # output layer
        self.phi = torch.tanh                                                     # activation function

    def forward(self, x):
        dx = (1 - torch.tanh(self.hidden(x))**2)
        dx2 = 2*torch.tanh(self.hidden(x))*(torch.tanh(self.hidden(x))**2 - 1)
        x = self.phi(self.hidden(x))                                              # output of hidden layer
        y = self.predict(x)                                                       # linear output
        return x, y, dx, dx2

class rnn_sin_exact_1d(torch.nn.Module):   
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = torch.sin

    def forward(self, x):
        dx = torch.cos(self.hidden(x))
        dx2 = -torch.sin(self.hidden(x))
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        return x, y, dx, dx2

class rnn_gauss_exact_1d(torch.nn.Module):   
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = gaussian

    def forward(self, x):
        dx = -self.hidden(x)*gaussian(self.hidden(x))
        dx2 = (self.hidden(x)**2 - 1)*gaussian(self.hidden(x))
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        return x, y, dx, dx2

#     def __init__(self, n_feature, n_hidden, n_output):
#         super().__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
#         self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
#         self.phi = gaussian
#         self.n_hidden = n_hidden
#
#     def forward(self, x, a, b, c, d):
#         xac = torch.zeros_like(x)
#         xac[:, 0:1] = a
#         xac[:, 1:2] = c
#         xac[:, 2:3] = x[:, 2:3]
#         xad = torch.zeros_like(x)
#         xad[:, 0:1] = a
#         xad[:, 1:2] = d
#         xad[:, 2:3] = x[:, 2:3]
#         xbc = torch.zeros_like(x)
#         xbc[:, 0:1] = b
#         xbc[:, 1:2] = c
#         xbc[:, 2:3] = x[:, 2:3]
#         xbd = torch.zeros_like(x)
#         xbd[:, 0:1] = b
#         xbd[:, 1:2] = d
#         xbd[:, 2:3] = x[:, 2:3]
#         dx = -self.hidden(x)*gaussian(self.hidden(x))
#         dx2 = (self.hidden(x)**2 - 1)*gaussian(self.hidden(x))
#         x = self.phi(self.hidden(x)) - (int_2d_gaussian(self.hidden(xbd)) - int_2d_gaussian(self.hidden(xbc)) - int_2d_gaussian(self.hidden(xad)) + int_2d_gaussian(self.hidden(xac)))/(((self.hidden.weight.data[:, 0:1]).reshape([-1, self.n_hidden]))*((self.hidden.weight.data[:, 1:2]).reshape([-1, self.n_hidden])))
#         y = self.predict(x)
#         return x, y, dx, dx2

class rnn_gauss_positive_exact_1d(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = gaussian

    def forward(self, x):
        dx = -self.hidden(x)*gaussian(self.hidden(x))
        dx2 = (self.hidden(x)**2 - 1)*gaussian(self.hidden(x))
        x = self.phi(self.hidden(x))
        # y = relu_like(self.predict(x))
        y = torch.relu(self.predict(x))
        return x, y, dx, dx2

class rnn_gauss_positive1_exact_1d(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = gaussian

    def forward(self, x):
        dx = -self.hidden(x)*gaussian(self.hidden(x))
        dx2 = (self.hidden(x)**2 - 1)*gaussian(self.hidden(x))
        x = self.phi(self.hidden(x))
        y1 = self.predict(x)
        y = relu_like(y1)
        return x, y, dx, dx2

class rnn_tanh_exact_2d(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)            # output layer
        self.phi = torch.tanh                                                     # activation function

    def forward(self, x):
        dx = (1 - torch.tanh(self.hidden(x))**2)
        dx2 = 2*torch.tanh(self.hidden(x))*(torch.tanh(self.hidden(x))**2 - 1)
        x = self.phi(self.hidden(x))                                              # output of hidden layer
        y = self.predict(x)                                                       # linear output
        y1 = y[:,0:1]
        y2 = y[:,1:2]
        return x, y, y1, y2, dx, dx2

class rnn_sin_exact_2d_vec(torch.nn.Module):      
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 1
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output 1
        self.phi1 = torch.sin
        self.phi2 = torch.sin

    def forward(self, x):
        dx = torch.cos(self.hidden(x))
        dx2 = -torch.sin(self.hidden(x))
        x1 = self.phi1(self.hidden(x))
        x2 = self.phi2(self.hidden(x))
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        y = torch.cat((y1,y2),dim = 1)
        return y, x1, x2, y1, y2, dx, dx2

class rnn_tanh_exact_2d_vec(torch.nn.Module):      
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 1
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output 1
        self.phi1 = torch.tanh
        self.phi2 = torch.tanh

    def forward(self, x):
        dx = (1 - torch.tanh(self.hidden(x))**2)
        dx2 = 2*torch.tanh(self.hidden(x))*(torch.tanh(self.hidden(x))**2 - 1)
        x1 = self.phi1(self.hidden(x))
        x2 = self.phi2(self.hidden(x))
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        y = torch.cat((y1,y2),dim = 1)
        return y, x1, x2, y1, y2, dx, dx2

class rnn_gauss_exact_2d_vec(torch.nn.Module):      
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # active 1
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output 1
        self.phi1 = gaussian
        self.phi2 = gaussian

    def forward(self, x):
        dx = -self.hidden(x)*gaussian(self.hidden(x))
        dx2 = (self.hidden(x)**2 - 1)*gaussian(self.hidden(x))
        x1 = self.phi1(self.hidden(x))
        x2 = self.phi2(self.hidden(x))
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        y = torch.cat((y1,y2),dim = 1)
        return y, x1, x2, y1, y2, dx, dx2

def compute_nn_derivatives_1d(network, m, points):
    output = network(points)
    hidden_weights = network.hidden.weight.data
    d1, d2 = output[2], output[3]

    dx = hidden_weights[:, 0:1].reshape([-1, m]) * d1
    dy = hidden_weights[:, 1:2].reshape([-1, m]) * d1
    dt = hidden_weights[:, 2:3].reshape([-1, m]) * d1
    dxx = hidden_weights[:, 0:1].reshape([-1, m]) ** 2 * d2
    dyy = hidden_weights[:, 1:2].reshape([-1, m]) ** 2 * d2
    return output[0], d1, d2, dx, dy, dt, dxx, dyy

# def compute_nn_derivatives_21(network, m, points):
#     output = network(points)
#     hidden_weights = network.hidden.weight.data
#     d1, d2 = output[2], output[3]
#
#     dx = hidden_weights[:, 0:1].reshape([-1, m]) * d1
#     dy = hidden_weights[:, 1:2].reshape([-1, m]) * d1
#     dxx = hidden_weights[:, 0:1].reshape([-1, m]) ** 2 * d2
#     dyy = hidden_weights[:, 1:2].reshape([-1, m]) ** 2 * d2
#     return output[0], d1, d2, dx, dy, dxx, dyy

# def compute_nn_derivatives_32_df(network, m, points):
#     output = network(points)
#     hidden_weights = network.hidden.weight.data
#     d1, d2 = output[5], output[5]
#
#     u1mdx = hidden_weights[:, 0:1].reshape([-1, m]) * hidden_weights[:, 1:2].reshape([-1, m]) * d1
#     u1mdy = hidden_weights[:, 1:2].reshape([-1, m])**2 * d1
#     u1mdt = hidden_weights[:, 2:3].reshape([-1, m]) * hidden_weights[:, 1:2].reshape([-1, m]) * d1
#     u2mdx = -hidden_weights[:, 0:1].reshape([-1, m])**2 * d1
#     u2mdy = -hidden_weights[:, 0:1].reshape([-1, m]) * hidden_weights[:, 1:2].reshape([-1, m]) * d1
#     u2mdt = -hidden_weights[:, 0:1].reshape([-1, m]) * hidden_weights[:, 2:3].reshape([-1, m]) * d1
#     u1mdxx = hidden_weights[:, 0:1].reshape([-1, m])**2 * hidden_weights[:, 1:2].reshape([-1, m]) * d2
#     u1mdyy = hidden_weights[:, 1:2].reshape([-1, m])**3 * d2
#     u2mdxx = -hidden_weights[:, 0:1].reshape([-1, m])**3 * d2
#     u2mdyy = -hidden_weights[:, 0:1].reshape([-1, m]) * hidden_weights[:, 1:2].reshape([-1, m])**2 * d2
#     return u1mdx, u1mdy, u1mdt, u2mdx, u2mdy, u2mdt, u1mdxx, u1mdyy, u2mdxx, u2mdyy

def weights_init_uniform(m, r1, r2):
    nn.init.uniform_(m.hidden.weight,r1,r2) 
    nn.init.uniform_(m.hidden.bias, r1, r2) 
    nn.init.uniform_(m.predict.weight,r1,r2) 

def weights_init_uniform_0(m, r1, r2):
    nn.init.uniform_(m.hidden.weight,r1,r2) 
    nn.init.uniform_(m.hidden.bias, r1, r2) 
    nn.init.constant_(m.predict.weight, 0)

def bias_re_2d(m, n, a, b, c, d):
    bb = torch.zeros(n, 2)
    bb[:, 0:1] = (b - a) * torch.rand(n, 1) + a
    bb[:, 1:2] = (d - c) * torch.rand(n, 1) + c
    bbb = m.hidden.weight.data * bb
    m.hidden.bias.data = -(bbb[:, 0] + bbb[:, 1])

def bias_re_3d(m, n, a, b, c, d, e, f):
    bb = torch.zeros(n, 3)
    bb[:, 0:1] = (b - a) * torch.rand(n, 1) + a
    bb[:, 1:2] = (d - c) * torch.rand(n, 1) + c
    bb[:, 2:3] = (f - e) * torch.rand(n, 1) + e
    bbb = m.hidden.weight.data * bb
    m.hidden.bias.data = -(bbb[:, 0] + bbb[:, 1] + bbb[:, 2])

def curl_free_2d_sin(m, n):
    def sin_curl_1(x):
        x1 = torch.sin(x)
        x2 = (m.hidden.weight.data[:, 0:1]).reshape([-1, n]) * x1
        return x2

    def sin_curl_2(x):
        x1 = torch.sin(x)
        x2 = (m.hidden.weight.data[:, 1:2]).reshape([-1, n]) * x1
        return x2

    m.phi1 = sin_curl_1
    m.phi2 = sin_curl_2
    return m

    # tanh

def div_free_2d_tanh(m, n):
    def tanh_div_1(x):
        x1 = torch.tanh(x)
        x2 = (m.hidden.weight.data[:, 1:2]).reshape([-1, n]) * x1
        return x2

    def tanh_div_2(x):
        x1 = torch.tanh(x)
        x2 = - (m.hidden.weight.data[:, 0:1]).reshape([-1, n]) * x1
        return x2

    m.phi1 = tanh_div_1
    m.phi2 = tanh_div_2
    return m

def div_free_2d_sin(m, n):
    def sin_div_1(x):
        x1 = torch.sin(x)
        x2 = (m.hidden.weight.data[:, 1:2]).reshape([-1, n]) * x1
        return x2

    def sin_div_2(x):
        x1 = torch.sin(x)
        x2 = - (m.hidden.weight.data[:, 0:1]).reshape([-1, n]) * x1
        return x2

    m.phi1 = sin_div_1
    m.phi2 = sin_div_2
    return m

def div_free_2d_gauss(m, n):
    def gauss_div_1(x):
        x1 = gaussian(x)
        x2 = (m.hidden.weight.data[:, 1:2]).reshape([-1, n]) * x1
        return x2

    def gauss_div_2(x):
        x1 = gaussian(x)
        x2 = - (m.hidden.weight.data[:, 0:1]).reshape([-1, n]) * x1
        return x2

    m.phi1 = gauss_div_1
    m.phi2 = gauss_div_2
    return m

def optimize_network(model, points, lr=0.01, num_epochs=2000):
    """Optimized plotting helper."""

    model.train()

    optimizer = torch.optim.Adam(model.hidden.parameters(), lr=lr)

    losses = []
    progress_bar = tqdm(range(num_epochs), desc="""Optimized plotting helper.""")

    for epoch in progress_bar:
        optimizer.zero_grad()

        _, y, _, _ = model(points)

        p = torch.mean(y)

        # loss = torch.sum((y - p) ** 2)
        loss = abs(y - p)


        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mean': f'{p.item():.6f}'
            })

    model.eval()
    return model, losses

def identity(x):
    return x

def optimize_network_1(model, points, lr=0.01, num_epochs=2000):
    """Optimized plotting helper."""
    model.phi = identity

    model.train()

    optimizer = torch.optim.Adam(model.hidden.parameters(), lr=lr)

    losses = []
    progress_bar = tqdm(range(num_epochs), desc="""Optimized plotting helper.""")

    for epoch in progress_bar:
        optimizer.zero_grad()

        _, y, _, _ = model(points)

        p = torch.mean(y)

        loss = torch.sum((y - p) ** 2)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mean': f'{p.item():.6f}'
            })

    model.eval()
    return model, losses
