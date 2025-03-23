FROM pytorch/pytorch:latest
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir streamlit torchvision matplotlib pandas pillow scikit-learn
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
