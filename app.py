import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load training history
history = pd.read_csv("training_history.csv")

# Define model (same architecture as train.py)
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

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("🧠 MNIST Digit Classifier with Streamlit")
st.sidebar.header("Options")
page = st.sidebar.radio("Choose a feature", ["📈 Training Analytics", "✏️ Draw a Digit", "📤 Upload an Image"])

# svm training results (no epochs)
svm_training_time = 115200  # seconds
svm_accuracy = 95.80

nn_total_time = history["time"].sum()
nn_final_accuracy = history["accuracy"].iloc[-1]

if page == "📈 Training Analytics":
    st.header("📊 Training Loss, Accuracy & Time")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].plot(history["epoch"], history["loss"], marker='o', linestyle='-', color='b')
    ax[0].set_title("Training Loss Curve")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    ax[1].plot(history["epoch"], history["accuracy"], marker='o', linestyle='-', color='g')
    ax[1].set_title("Training Accuracy Curve")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy (%)")

    # Add horizontal lines for SVM results
    ax[1].axhline(y=svm_accuracy, color="green", linestyle="--", label="SVM Accuracy")

    ax[2].plot(history["epoch"], history["time"], marker='o', linestyle='-', color='b')
    ax[2].set_title("Training Time Curve")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Time (s)")

    st.pyplot(fig)

    # Compare total training time
    st.header("Compare Total Training Time")
    training_time_data = pd.DataFrame({
        "Model": ["NN", "SVM"],
        "Training Time (sec)": [nn_total_time, svm_training_time]
    })

    fig2, ax2 = plt.subplots()
    ax2.bar(training_time_data["Model"], training_time_data["Training Time (sec)"],
       color=["blue", "orange"])
    ax2.set_ylabel("Training Time (sec)")
    ax2.set_title("Training Time Comparison")
    st.pyplot(fig2)

    # compare final accuracy
    st.header("Compare Final Accuracy")
    accuracy_data = pd.DataFrame({
        "Model": ["NN", "SVM"],
        "Accuracy (%)": [nn_final_accuracy, svm_accuracy]
    })

    fig3, ax = plt.subplots()
    ax.bar(accuracy_data["Model"], accuracy_data["Accuracy (%)"],
       color=["blue", "orange"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Accuracy Comparison")
    st.pyplot(fig3)

elif page == "📤 Upload an Image":
    st.header("🖼 Upload a Digit Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        st.write(f"📝 Prediction: **{predicted.item()}**")
