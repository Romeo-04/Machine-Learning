import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Check for and handle NaN values
print("Checking for NaN values...")
print(f"Train data NaN count: {train.isna().sum().sum()}")
print(f"Test data NaN count: {test.isna().sum().sum()}")

# Remove rows with NaN values
train = train.dropna()
test = test.dropna()

print(f"After cleaning - Train shape: {train.shape}, Test shape: {test.shape}")

X_test = test["x"].values.reshape(-1, 1)
y_test = test["y"].values
X_train = train["x"].values.reshape(-1, 1)
y_train = train["y"].values

# Convert continuous y values to discrete classes for classification
# This creates classes based on value ranges
def create_classes(y_values, n_classes=5):  # Increased to 5 classes for better distribution
    """Convert continuous values to discrete classes"""
    min_val, max_val = y_values.min(), y_values.max()
    thresholds = np.linspace(min_val, max_val, n_classes + 1)
    classes = np.digitize(y_values, thresholds) - 1
    classes = np.clip(classes, 0, n_classes - 1)  # Ensure values are in valid range
    return classes

# Convert y values to classes
y_train_classes = create_classes(y_train)
y_test_classes = create_classes(y_test)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train_classes)

y_pred = knn.predict(X_test)

print("Confusion Matrix: ", confusion_matrix(y_test_classes, y_pred))
print("Classification Report: ", classification_report(y_test_classes, y_pred))

print("="*50)
print("OPTION 1: REGRESSION APPROACH (Recommended)")
print("="*50)
# Since your data is continuous, regression is more appropriate
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)

y_pred_reg = knn_reg.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_reg):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_reg):.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('KNN Regression: Actual vs Predicted')
plt.show()

print("\n" + "="*50)
print("OPTION 2: CLASSIFICATION APPROACH")
print("="*50)
