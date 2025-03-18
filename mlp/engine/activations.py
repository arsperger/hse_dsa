import numpy as np
from .tensor import Tensor

def relu(x):
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    def _backward():
        x.grad += (x.data > 0) * out.grad

    out._backward = _backward
    out._prev = {x}
    return out

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(sig, requires_grad=x.requires_grad)

    def _backward():
        x.grad += sig * (1 - sig) * out.grad

    out._backward = _backward
    out._prev = {x}
    return out
