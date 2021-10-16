import math
import needle as ndl

from .ops import *

def uniform(x, low=0.0, high=1.0):
    x.data = randu(x.shape, low=low, high=high, dtype=x.dtype, device=x.device).data

def normal(x, mean=0.0, std=1.0):
    x.data = randn(x.shape, mean=mean, std=std, dtype=x.dtype, device=x.device).data

def constant(x, c=0.0):
    x.data = full(x.shape, c, dtype=x.dtype, device=x.device).data

def ones(x):
    constant(x, c=1.0)

def zeros(x):
    constant(x, c=0.0)

def xavier_uniform(x, gain=1.0):
    a = gain * math.sqrt(6.0/sum(x.shape[-2:]))
    uniform(x, low=-a, high=a)

def xavier_normal(x, gain=1.0):
    a = gain * math.sqrt(2.0/sum(x.shape[-2:]))
    normal(x, mean=0, std=a)

def kaiming_uniform(x, mode='fan_in', nonlinearity='relu'):
    gain = 1.0
    if nonlinearity == 'relu':
        gain = math.sqrt(2)
    if mode == 'fan_in':
        fan_mode = x.shape[-2]
    else:
        fan_mode = x.shape[-1]
    bound = gain * math.sqrt(3.0/fan_mode)
    uniform(x, low=-bound, high=bound)

def kaiming_normal(x, mode='fan_in', nonlinearity='relu'):
    gain = 1.0
    if nonlinearity == 'relu':
        gain = math.sqrt(2)
    if mode == 'fan_in':
        fan_mode = x.shape[-2]
    else:
        fan_mode = x.shape[-1]
    std = gain / math.sqrt(fan_mode)
    normal(x, mean=0, std=std)


def _calculate_fans(x):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION
