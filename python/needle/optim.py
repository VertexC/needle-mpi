"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.delta = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.requires_grad:
                detached_param_data = param.data
                new_grad = param.grad.data + self.weight_decay*detached_param_data
                if param not in self.delta:
                    self.delta[param] = new_grad
                else:
                    self.delta[param] = self.momentum * self.delta[param] + new_grad
                grad = self.delta[param]
                param.data = detached_param_data - self.lr*grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.requires_grad:
                detached_param_data = param.data
                new_grad = param.grad.data + self.weight_decay*detached_param_data
                if i not in self.u:
                    self.u[i] = (1-self.beta1)*new_grad
                else:
                    self.u[i] = self.beta1*self.u[i] + (1-self.beta1)*new_grad
                
                if i not in self.v:
                    self.v[i] = (1-self.beta2)*new_grad**2
                else:
                    self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(new_grad**2)
                if self.bias_correction:
                    u_hat = self.u[i] / (1-self.beta1**self.t)
                    v_hat = self.v[i] / (1-self.beta2**self.t)
                else:
                    u_hat = self.u[i]
                    v_hat = self.v[i]
                print(i, param.grad.data.numpy().flatten()[:10])
                print(i, param.data.numpy().flatten()[:10])
                print(i, u_hat.numpy().flatten()[:10])
                print(i, v_hat.numpy().flatten()[:10])
                print()
                param.data = detached_param_data - self.lr*u_hat/(v_hat**0.5+self.eps)
