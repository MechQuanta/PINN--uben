import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io

class NN(torch.nn.Module):
    def __init__(self,layers):
        super(NN,self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
    def forward(self,X):
        for i in range(len(self.layers)-1):
            X = torch.nn.Tanh(self.layers[i](X))
        X = self.Linear[-1](X)
        return X
def net_u(model, X):
    X = model(X)
    return X


def net_f(model, X, nu):
    X.requires_grad = True
    u = model(X)
    u_X = torch.autograd.grad(
        u,
        X,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    u_x = u_X[:, 0]
    u_t = u_X[:, 1]
    u_XX = torch.autograd.grad(
        u_X,
        X,
        grad_outputs=True,
        create_graph=True,
        retain_graph=True
    )[0]
    u_xx = u_XX[:, 0]
    f = u_t + u.squeeze() * u_x - nu * u_xx
    return f

data = scipy.io.loadmat('./burgers_shock.mat')
print(data['t'])
