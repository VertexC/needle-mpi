"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy as np
from .autograd import Op, Tensor, Value, Tuple
from .device import default_device
import copy

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]

multiply_scalar = register_op("MulScalar", MulScalarOp())


class PowerScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        scalar = node.attrs["scalar"]
        return [out_grad * scalar * power_scalar(node.inputs[0], scalar-1)]
        
power_scalar = register_op("PowerScalar", PowerScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        x, y = node.inputs
        return [out_grad * ones_like(x) / y, -out_grad * x / (y*y)]


divide = register_op("EWiseDiv", EWiseDivOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return [out_grad * ones_like(x) / node.attrs["scalar"]]


divide_scalar = register_op("DivScalar", DivScalarOp())


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        x, y = node.inputs
        diff = len(x.shape) - len(y.shape)
        if diff < 0:
            return [summation(matmul(out_grad, transpose(y)), axes=tuple(np.arange(-diff).tolist())), matmul(transpose(x), out_grad)]
        elif diff > 0:
            return [matmul(out_grad, transpose(y)), summation(matmul(transpose(x), out_grad), axes=tuple(np.arange(diff).tolist()))]
        else:
            return [matmul(out_grad, transpose(y)), matmul(transpose(x), out_grad)]


matmul = register_op("MatMul", MatMulOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        temp_shape = list(node.inputs[0].shape)
        if node.attrs["axes"] is None:
            temp_shape = [1]
        else:
            for axis in node.attrs["axes"]:
                temp_shape[axis] = 1
        return [broadcast_to(reshape(out_grad, tuple(temp_shape)), node.inputs[0].shape)]


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        out_shape = out_grad.shape
        in_shape = node.inputs[0].shape
        reduce_sum_axes = []
        i = 1
        while i < len(in_shape)+1:
            if out_shape[-i] > in_shape[-i]:
                reduce_sum_axes.append(-i)
            i += 1
                
        for j in range(0, len(out_shape)-i+1):
            reduce_sum_axes.append(j)
        print("broadcast reshape", summation(out_grad, axes=tuple(reduce_sum_axes)).shape, in_shape)
        result = reshape(summation(out_grad, axes=tuple(reduce_sum_axes)), in_shape)
        return [result]

broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        return [reshape(out_grad, node.inputs[0].shape)]


reshape = register_op("Reshape", ReshapeOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [out_grad * negate(ones_like(node.inputs[0]))]


negate = register_op("Negate", NegateOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        return [transpose(out_grad, node.attrs["axes"])]

transpose = register_op("Transpose", TransposeOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return [out_grad * ones_like(x) / x]


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())

class ReLuGradOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [zeros_like(node.inputs[0])]

relu_grad = register_op("ReLUGrad", ReLuGradOp())

class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [out_grad * relu_grad(node.inputs[0])]

relu = register_op("ReLU", ReLUOp())

class LogSoftmaxOp(Op):
    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [x])

    # reference: https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        E_x = exp(logsoftmax(x))
        return [out_grad - E_x * reshape(summation(out_grad, axes=tuple([-1])), tuple([-1,1]))]

logsoftmax = register_op("LogSoftmax", LogSoftmaxOp())

# additional helper functions
def full(shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False):
    device = device if device else default_device()
    
    if not rand or 'dist' not in rand:
        arr = device.empty(shape, dtype)
        device.fill(arr, fill_value)
    else:
        if rand['dist'] == 'normal':
            arr = device.randn(shape, dtype, mean=rand['mean'], std=rand['std'])
        if rand['dist'] == 'binomial':
            arr = device.randb(shape, dtype, ntrials=rand['trials'], p=rand['prob'])
        if rand['dist'] == 'uniform':
            arr = device.randu(shape, dtype, low=rand['low'], high=rand['high'])
            
    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def one_hot(labels: Tensor, *, num_classes=10, dtype="float32", device=None):
    device = device if device else default_device()
    arr = device.one_hot(labels.numpy(), num_classes=num_classes)
    return Tensor.make_const(arr, device, requires_grad=False)

def mean(a:Tensor, *, axes: Optional[tuple] = None, keep_dim=False):
    divide_scalar = 1
    
    if axes is None:
        axes = [i for i in range(len(a.shape))]
    for axis in axes:
        divide_scalar *= a.shape[axis]
    out = summation(a, axes=axes) / divide_scalar
    if keep_dim:
        keep_shape = list(a.shape)
        for axis in axes:
            keep_shape[axis] = 1
        out = reshape(out, tuple(keep_shape))
    return out

def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)

def ones(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 1, dtype=dtype, device=device, requires_grad=requires_grad)

def randn(shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'normal', 'mean': mean, 'std': std}, dtype=dtype, device=device, requires_grad=requires_grad)


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'binomial', 'trials': n, 'prob': p}, dtype=dtype, device=device, requires_grad=requires_grad)

def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'uniform', 'low': low, 'high': high}, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )