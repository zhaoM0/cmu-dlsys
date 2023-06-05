"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # raise NotImplementedError()
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        # raise NotImplementedError()
        in_mat = node.inputs[0]
        return self.scalar * out_grad * Tensor(array_api.power(in_mat, self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-lhs / (rhs * rhs))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        assert in_shape == out_grad.shape 
        return Tensor(array_api.ones(in_shape) / self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # raise NotImplementedError()
        _axis = list(range(0, a.ndim))
        
        if self.axes is not None:
            _axis[self.axes[0]], _axis[self.axes[1]] = _axis[self.axes[1]], _axis[self.axes[0]] 
        else:
            _axis[-2], _axis[-1] = _axis[-1], _axis[-2]  

        return array_api.transpose(a, _axis)
        

    def gradient(self, out_grad, node):
        # raise NotImplementedError()
        in_shape = node.inputs[0].shape
        return out_grad.reshape(in_shape)

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)  

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        return out_grad.reshape(in_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        in_shape  = node.inputs[0].shape
        out_shape = out_grad.shape 
        _axis     = []
        
        if len(in_shape) == len(out_shape):
            _axis = [ loc for loc, val in enumerate(in_shape) if val == 1 ]
        else:
            _axis = list(range(len(out_shape)))
            
        return out_grad.sum(tuple(_axis)).reshape(in_shape)
    

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        in_shape  = list(node.inputs[0].shape)
        out_shape = in_shape.copy()
        
        if not self.axes:
            return out_grad.broadcast_to(in_shape)
        else:
            for i in self.axes:
                out_shape[i] = 1
            return out_grad.reshape(out_shape).broadcast_to(in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lmat, rmat = node.inputs
        
        grad_lmat = out_grad @ rmat.transpose()
        grad_rmat = lmat.transpose() @ out_grad
        
        if grad_lmat.shape != lmat.shape:
            _axis = range(len(grad_lmat.shape) - len(lmat.shape))
            grad_lmat = grad_lmat.sum(tuple(_axis))
            
        if grad_rmat.shape != rmat.shape:
            _axis = range(len(grad_rmat.shape) - len(rmat.shape))
            grad_rmat = grad_rmat.sum(tuple(_axis))
            
        assert grad_lmat.shape == lmat.shape
        assert grad_rmat.shape == rmat.shape
        
        return grad_lmat, grad_rmat


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        # raise NotImplementedError()
        return -out_grad

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        in_mat = node.inputs[0].numpy()
        return out_grad * Tensor(array_api.where(in_mat > 0, 1, 0))


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # add your solution
        out_shape = array_api.amax(Z, axis = self.axes).shape 
        max_z = array_api.amax(Z, axis = self.axes, keepdims = True)
        out_mat = array_api.log(
                    array_api.sum(array_api.exp(Z - max_z), axis = self.axes, keepdims = True)
                  ) + max_z
        return out_mat.reshape(out_shape)
        

    def gradient(self, out_grad, node):
        # add your solution
        raise NotImplementedError()
        in_mat = node.inputs[0]
        
        
        

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
