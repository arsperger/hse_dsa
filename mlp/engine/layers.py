from .tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias
