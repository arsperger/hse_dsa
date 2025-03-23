import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Load MNIST data
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Split data (using a smaller subset for faster grid search)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=20000, test_size=2000, random_state=42, stratify=y
)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'C': [1, 10],
    'gamma': [0.01, 0.001],
    'kernel': ['rbf']  # RBF usually works best for MNIST
}

# Create and run grid search
print("Starting grid search (this may take a while)...")
grid_search = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Train the model
grid_search.fit(X_train_scaled, y_train)

# Print results
print("\nGrid search complete!")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get best model
best_svm = grid_search.best_estimator_

# Evaluate on test set with inference time measurement
print("\nEvaluating on test set...")
inference_start = time.time()
y_pred = best_svm.predict(X_test_scaled)
inference_time = time.time() - inference_start

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Inference time on test set: {inference_time:.4f} seconds")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted digit')
plt.ylabel('True digit')
plt.savefig("confusion_matrix.png")

# Plot sample predictions
def plot_samples(images, labels, predictions, n_samples=10):
    plt.figure(figsize=(15, 4))
    indices = np.random.choice(range(len(images)), n_samples, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i + 1)
        image = images[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {labels[idx]}\nPred: {predictions[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

# Convert to numpy arrays if needed
X_test_array = np.array(X_test)
y_test_array = np.array(y_test)

print("Plotting prediction samples...")
plot_samples(X_test_array, y_test_array, y_pred, n_samples=10)

# Save the model, scaler, and training history
print("Saving model and results...")
joblib.dump(best_svm, "svm_best_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
joblib.dump(grid_search.cv_results_, "grid_search_results.pkl")

# Create a dictionary to store key metrics including inference time
training_history = {
    "best_params": [str(grid_search.best_params_)],
    "best_cv_score": [grid_search.best_score_],
    "test_accuracy": [accuracy],
    "inference_time": [inference_time]
}
df_history = pd.DataFrame(training_history)
df_history.to_csv("training_history.csv", index=False)

print("Model and training history saved.")
