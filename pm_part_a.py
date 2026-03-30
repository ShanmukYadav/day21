"""
Assignment — Week 04 · Day 21 (PM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part A — Concept Application (40%)
Topics: Regression, Classification, Bias-Variance Tradeoff
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_regression, make_classification

print("="*80)
print("PART A: CONCEPT APPLICATION (40%)")
print("="*80)

# ============================================================================
# 1. Create synthetic datasets
# ============================================================================
print("\n1. CREATING SYNTHETIC DATASETS")
print("-" * 80)

# Regression dataset (continuous target)
print("Regression Dataset (Continuous Target):")
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
print(f"  Shape of X: {X_reg.shape}")
print(f"  Shape of y: {y_reg.shape}")
print(f"  First 5 X values: {X_reg[:5].flatten()}")
print(f"  First 5 y values: {y_reg[:5]}")
print(f"  y range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

# Classification dataset (binary target)
print("\nClassification Dataset (Binary Target):")
X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_informative=2, 
                                   n_redundant=0, random_state=42)
print(f"  Shape of X: {X_clf.shape}")
print(f"  Shape of y: {y_clf.shape}")
print(f"  Class distribution: {np.bincount(y_clf)}")
print(f"  Unique classes: {np.unique(y_clf)}")

# ============================================================================
# 2a. Train regression model and visualize
# ============================================================================
print("\n2a. REGRESSION MODEL - Linear Regression")
print("-" * 80)

# Train linear regression
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Make predictions
y_pred_reg = reg_model.predict(X_reg)

# Calculate MSE
mse = mean_squared_error(y_reg, y_pred_reg)

print(f"Model Coefficient: {reg_model.coef_[0]:.4f}")
print(f"Model Intercept: {reg_model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

# Print some predictions vs actual
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"  X={X_reg[i, 0]:6.2f} | Actual y={y_reg[i]:8.2f} | Predicted y={y_pred_reg[i]:8.2f}")

# Plot regression
plt.figure(figsize=(10, 5))
plt.scatter(X_reg, y_reg, color='blue', label='Actual data', alpha=0.6)
plt.plot(X_reg, y_pred_reg, color='red', linewidth=2, label='Regression line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression: Fitting a Line to Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/claude/regression_plot.png', dpi=100, bbox_inches='tight')
print("\n✓ Regression plot saved: regression_plot.png")
plt.close()

# ============================================================================
# 2b. Train classification model and visualize
# ============================================================================
print("\n2b. CLASSIFICATION MODEL - Logistic Regression")
print("-" * 80)

# Train logistic regression
clf_model = LogisticRegression(random_state=42)
clf_model.fit(X_clf, y_clf)

# Make predictions
y_pred_clf = clf_model.predict(X_clf)

# Calculate accuracy
accuracy = accuracy_score(y_clf, y_pred_clf)

print(f"Model Coefficients: {clf_model.coef_[0]}")
print(f"Model Intercept: {clf_model.intercept_[0]:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Print some predictions vs actual
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"  X={X_clf[i]} | Actual class={y_clf[i]} | Predicted class={y_pred_clf[i]}")

# Plot classification
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='viridis', 
                     edgecolors='k', s=100, alpha=0.7, label='Actual')

# Plot decision boundary
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = clf_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.xlabel('Feature 1 (X1)')
plt.ylabel('Feature 2 (X2)')
plt.title(f'Logistic Regression Classification (Accuracy: {accuracy:.2%})')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.savefig('/home/claude/classification_plot.png', dpi=100, bbox_inches='tight')
print("✓ Classification plot saved: classification_plot.png")
plt.close()

# ============================================================================
# 3. Identify problem type and justify
# ============================================================================
print("\n3. PROBLEM IDENTIFICATION")
print("-" * 80)

problem_examples = [
    ("Predicting house prices", "REGRESSION", 
     "Target is continuous (price in $)"),
    ("Predicting if email is spam", "CLASSIFICATION", 
     "Target is binary (spam/not spam)"),
    ("Predicting customer lifetime value", "REGRESSION", 
     "Target is continuous (monetary value)"),
    ("Predicting disease presence", "CLASSIFICATION", 
     "Target is binary (disease/no disease)"),
    ("Predicting stock price movement", "REGRESSION", 
     "Target is continuous (price value)"),
]

print("Examples and Problem Type Classification:")
for example, problem_type, justification in problem_examples:
    print(f"\n  Problem: {example}")
    print(f"  Type: {problem_type}")
    print(f"  Justification: {justification}")

# ============================================================================
# 4. Implement simple regression model manually
# ============================================================================
print("\n4. MANUAL REGRESSION IMPLEMENTATION")
print("-" * 80)

print("Simple Linear Regression: y = mx + b")
print("Formula: m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)")
print("         b = ȳ - m*x̄")

# Create simple dataset
X_manual = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_manual = np.array([2, 4, 5, 4, 6])

print(f"\nDataset:")
print(f"  X: {X_manual.flatten()}")
print(f"  y: {y_manual}")

# Manual calculation
x_mean = np.mean(X_manual)
y_mean = np.mean(y_manual)

numerator = np.sum((X_manual.flatten() - x_mean) * (y_manual - y_mean))
denominator = np.sum((X_manual.flatten() - x_mean) ** 2)

m = numerator / denominator
b = y_mean - m * x_mean

print(f"\nCalculated Parameters:")
print(f"  Slope (m): {m:.4f}")
print(f"  Intercept (b): {b:.4f}")
print(f"  Equation: y = {m:.4f}*x + {b:.4f}")

# Predictions
y_pred_manual = m * X_manual.flatten() + b
print(f"\nPredictions:")
for i in range(len(X_manual)):
    print(f"  x={X_manual[i, 0]} | actual y={y_manual[i]} | predicted y={y_pred_manual[i]:.4f}")

# Calculate MSE manually
mse_manual = np.mean((y_manual - y_pred_manual) ** 2)
print(f"\nMean Squared Error (MSE): {mse_manual:.4f}")

# ============================================================================
# 5. Implement simple classification logic
# ============================================================================
print("\n5. MANUAL CLASSIFICATION IMPLEMENTATION")
print("-" * 80)

print("Simple Threshold-based Classification")
print("Rule: if score >= threshold → class 1, else class 0")

# Generate some scores
scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.5, 0.7, 0.3, 0.95])
actual_labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 1])
threshold = 0.5

print(f"\nScores: {scores}")
print(f"Actual Labels: {actual_labels}")
print(f"Threshold: {threshold}")

# Classify
predicted_labels = (scores >= threshold).astype(int)

print(f"\nClassifications:")
print(f"{'Score':<8} {'Actual':<8} {'Predicted':<12} {'Correct':<8}")
print("-" * 40)
for i in range(len(scores)):
    correct = "✓" if predicted_labels[i] == actual_labels[i] else "✗"
    print(f"{scores[i]:<8.1f} {actual_labels[i]:<8} {predicted_labels[i]:<12} {correct:<8}")

# Calculate accuracy
accuracy_manual = np.mean(predicted_labels == actual_labels)
correct_count = np.sum(predicted_labels == actual_labels)
total_count = len(actual_labels)

print(f"\nAccuracy Calculation:")
print(f"  Correct predictions: {correct_count}/{total_count}")
print(f"  Accuracy: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")

# ============================================================================
# 6. Compare regression vs classification
# ============================================================================
print("\n6. REGRESSION vs CLASSIFICATION COMPARISON")
print("-" * 80)

comparison = """
ASPECT              | REGRESSION           | CLASSIFICATION
--------------------|----------------------|------------------------
Output Type        | Continuous values    | Discrete classes/labels
Example Output     | 25.5, 100.2, -5.1   | Class 0, Class 1, Cat, Dog
Use Cases          | Price prediction     | Email spam detection
                   | Temperature forecast | Disease diagnosis
                   | Stock prices         | Image recognition
                   | House values         | Sentiment analysis
--------------------|----------------------|------------------------
Evaluation         | MSE, RMSE, MAE       | Accuracy, Precision
Metrics            | R² Score             | Recall, F1-Score
                   |                      | Confusion Matrix
--------------------|----------------------|------------------------
Models             | Linear Regression    | Logistic Regression
Examples           | Polynomial Reg.      | Decision Trees
                   | SVR                  | Random Forest
                   | Neural Networks      | Neural Networks
                   | Gradient Boosting    | Gradient Boosting
--------------------|----------------------|------------------------
Output Range       | Any real number      | Fixed set of classes
Prediction Type    | How much/how many    | What category/class
"""

print(comparison)

print("\n" + "="*80)
print("PART A COMPLETED SUCCESSFULLY")
print("="*80)
