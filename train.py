import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load test dataset (for inference time measurement)
#testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
#testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load test dataset (for inference time measurement)
full_testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create a subset of 2000 samples
from torch.utils.data import Subset
import numpy as np

# For stratified sampling (balanced class distribution)
# Get all indices and labels
indices = list(range(len(full_testset)))
labels = [full_testset[i][1] for i in indices]

# Use sklearn's train_test_split for stratified sampling
from sklearn.model_selection import train_test_split

# Select 2000 samples while maintaining class distribution
_, subset_indices = train_test_split(
    indices,
    test_size=2000,  # Select exactly 2000 samples
    stratify=labels,  # Maintain class distribution
    random_state=42   # For reproducibility
)

# Create the subset and dataloader
testset = Subset(full_testset, subset_indices)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print(f"Test dataset size: {len(testset)} samples")

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = {"epoch": [], "loss": [], "accuracy": [], "train_time": [], "inference_time": []}

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = 100 * correct / total
    train_time = time.time() - start_time

    # Inference phase: measure time on test dataset
    model.eval()
    inference_start = time.time()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    inference_time = time.time() - inference_start
    model.train()

    history["epoch"].append(epoch + 1)
    history["loss"].append(epoch_loss)
    history["accuracy"].append(epoch_accuracy)
    history["train_time"].append(train_time)
    history["inference_time"].append(inference_time)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

print("Training Completed!")

# Save model and history
torch.save(model.state_dict(), "model.pth")
pd.DataFrame(history).to_csv("training_history.csv", index=False)
print("Model and training history saved.")
