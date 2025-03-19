import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        self.grad = np.ones_like(self.data)
        to_visit = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                to_visit.append(t)

        build_topo(self)
        for node in reversed(to_visit):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rsub__(self, other):
        return (other * 1).__sub__(self)

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(power * self.data ** (power - 1) * out.grad, self.data.shape)

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

def _unbroadcast(grad, shape):
    """
    Приводит градиент grad к заданной форме shape,
    суммируя по лишним осям, если это необходимо.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
