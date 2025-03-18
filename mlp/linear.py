import numpy as np
from engine.tensor import Tensor
from engine.layers import Linear
from engine.loss import mse_loss
from engine.optim import SGD

# Данные
np.random.seed(42)
x_data = np.random.rand(100, 1) * 10
y_data = 3 * x_data + 2 + np.random.randn(100, 1) * 2

x = Tensor(x_data, requires_grad=False)
y = Tensor(y_data, requires_grad=False)

# Модель
model = Linear(1, 1)
optimizer = SGD([model.weights, model.bias], lr=0.01)

# Обучение
epochs = 100
for epoch in range(epochs):
    y_pred = model(x)
    loss = mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

print(f"Final Weights: {model.weights.data}")
print(f"Final Bias: {model.bias.data}")
