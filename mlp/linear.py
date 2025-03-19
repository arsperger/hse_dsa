import numpy as np
from engine.tensor import Tensor
from engine.layers import Linear
from engine.loss import mse_loss
from engine.optim import SGD

# Генерируем данные: x_data от 0 до 10, y_data = 3*x + 2 + шум
np.random.seed(42)
x_data = np.random.rand(100, 1) * 10
y_data = 3 * x_data ** 2 + 2 + np.random.randn(100, 1) * 2

x = Tensor(x_data, requires_grad=False)
y = Tensor(y_data, requires_grad=False)

# модель и оптимизатор
lvl1 = Linear(1, 10)
lvl2 = Linear(10, 1)
model = lambda x: lvl2(lvl1(x))
parameters = [
    lvl1.weights, lvl1.bias,
    lvl1.weights, lvl1.bias,
]
optimizer = SGD(parameters, lr=0.001)

epochs = 100000
for epoch in range(epochs):
    y_pred = model(x)
    loss = mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}, Parameters: {[parameter.data.tolist() for parameter in parameters]}")

print(f"Final Parameters: {[parameter.data.tolist() for parameter in parameters]}")
