FROM pytorch/pytorch:latest
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torchvision matplotlib
CMD ["python", "train_mnist.py"]
