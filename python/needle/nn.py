"""The module.
"""
from __future__ import annotations
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters.
    """

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

def _child_modules(value: object) -> List[Module]:
    child_modules = []
    print(value)
    for k, v in value.items():
        if isinstance(v, Module):
            child_modules.append(v)
            child_modules += v._children()
        if isinstance(v, list) or isinstance(v, tuple):
            for a in v:
                if isinstance(a, Module):
                    child_modules.append(a)
                    child_modules += a._children()
        if isinstance(v, dict):
            child_modules += _child_modules(v)
    return child_modules


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module.
        """
        return _unpack_params(self.__dict__)

    def _children(self) -> List[Module]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = np.sqrt(1.0/in_features)
        self.weight = ops.randu((in_features, out_features), low=-k, high=k, dtype=dtype, device=device, requires_grad=True)
        self.bias = ops.randu(out_features, low=-k, high=k, dtype=dtype, device=device, requires_grad=True)

    def forward(self, x: Tensor)-> Tensor:
        return ops.matmul(x, self.weight) + self.bias  


class ReLU(Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules, device=None, dtype="float32"):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

class SoftmaxLoss(Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        num_classes = x.shape[-1]
        samples = x.shape[0]
        return -ops.summation(ops.multiply(ops.logsoftmax(x), ops.one_hot(y, num_classes=num_classes))) / samples


class BatchNorm(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = ops.ones(dim, dtype=dtype, device=device, requires_grad=True) 
        self.bias = ops.zeros(dim, dtype=dtype, device=device, requires_grad=True) 
        
        self.running_mean = ops.zeros(dim, dtype=dtype, device=device, requires_grad=False) 
        self.running_var = ops.ones(dim, dtype=dtype, device=device, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        axes = tuple([i for i in range(len(x.shape)) if i != 1])
        x_mean = ops.mean(x, axes=axes, keep_dim=True)
        eval_diff = x - ops.broadcast_to(x_mean, x.shape)
        x_var = ops.mean(eval_diff**2, axes=axes, keep_dim=True)

        weight = ops.broadcast_to(ops.reshape(self.weight, x_mean.shape), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, x_mean.shape), x.shape)
        if self.training:
            out = (weight * eval_diff) / ops.broadcast_to(ops.power_scalar(x_var + self.eps, 0.5), eval_diff.shape) + bias
        else:
            train_diff = x - ops.broadcast_to(ops.reshape(self.running_mean, x_mean), x.shape)
            out = (weight * train_diff) / ops.broadcast_to(ops.power_scalar(ops.reshape(self.running_var, x_var) + self.eps, 0.5), train_diff.shape) + bias
            
        
        self.running_mean = self.running_mean * (1-self.momentum) + self.momentum*ops.reshape(x_mean, self.running_mean.shape)  
        
        dims = 1
        for axis in axes:
            dims *= x.shape[axis]
        running_var = x_var * dims / (dims-1)
        self.running_var = self.running_var * (1-self.momentum) + self.momentum*ops.reshape(running_var, self.running_var.shape)  
        return out
        
        


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = ops.ones(dims, dtype=dtype, device=device, requires_grad=True) 
        self.bias = ops.zeros(dims, dtype=dtype, device=device, requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        z = x
        z_hat = z * self.weight + self.bias
        axes = tuple([-1-i for i in range(len(self.dims))])
        z_mean = ops.mean(z_hat, axes=axes, keep_dim=True)
        diff = z_hat - ops.broadcast_to(z_mean, z_hat.shape)
        z_var = ops.mean(diff**2, axes=axes, keep_dim=True)
        out = diff / ops.broadcast_to(ops.power_scalar(z_var + self.eps, 0.5), diff.shape)
        return out


class Dropout(Module):
    def __init__(self, drop_prob, device=None, dtype="float32"):
        super().__init__()
        self.p = drop_prob
        self.eps = 0.000000001
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x * (ops.ones_like(x, device=self.device)-ops.randb(x.shape, n=1, p=self.p, dtype=self.dtype, device=self.device)) / (1.0 - self.p + self.eps)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module, device=None, dtype="float32"):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Identity(Module):
    def __init__(self, *args, device=None, dtype="float32", **kwargs):
        super().__init__()

    def forward(self, x):
        return x
