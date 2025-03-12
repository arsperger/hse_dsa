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

if page == "📈 Training Analytics":
    st.header("📊 Training Loss & Accuracy")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(history["epoch"], history["loss"], marker='o', linestyle='-', color='b')
    ax[0].set_title("Training Loss Curve")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    ax[1].plot(history["epoch"], history["accuracy"], marker='o', linestyle='-', color='g')
    ax[1].set_title("Training Accuracy Curve")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy (%)")

    st.pyplot(fig)

elif page == "✏️ Draw a Digit":
    st.header("🎨 Draw a Digit Below")
    canvas = st.canvas(draw_mode="freedraw", height=200, width=200)

    if st.button("Predict"):
        img = canvas.image_data  # Get drawn image
        img = Image.fromarray(np.uint8(img)).convert("L")  # Convert to grayscale
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        st.write(f"📝 Prediction: **{predicted.item()}**")

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
