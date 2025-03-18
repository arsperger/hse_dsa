from .tensor import Tensor

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum()
