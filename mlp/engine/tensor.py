import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        to_visit = [self]
        visited = set()
        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)
            node._backward()
            to_visit.extend(node._prev)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data @ other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __pow__(self, power):
        return Tensor(self.data ** power, requires_grad=self.requires_grad)


    def sum(self):
        return Tensor(self.data.sum(), requires_grad=self.requires_grad)