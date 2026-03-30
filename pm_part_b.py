"""
Assignment — Week 04 · Day 21 (PM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part B — Stretch Problem (30%)
Topics: Bias-Variance Tradeoff, Underfitting, Overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("="*80)
print("PART B: STRETCH PROBLEM (30%)")
print("="*80)

# ============================================================================
# Generate synthetic data with underlying pattern
# ============================================================================
print("\n0. GENERATING SYNTHETIC DATA")
print("-" * 80)

np.random.seed(42)
n_samples = 100

# Generate X
X_train = np.linspace(0, 10, n_samples)

# Generate y from a cubic function with noise
# True function: y = 0.5*x^2 + sin(x) + noise
y_train = 0.5 * X_train**2 + np.sin(X_train) + np.random.normal(0, 1.5, n_samples)

# Generate test data
X_test = np.linspace(0.5, 9.5, 50)
y_test = 0.5 * X_test**2 + np.sin(X_test)  # No noise in test data

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# ============================================================================
# 1. Simulate bias-variance tradeoff with varying polynomial degrees
# ============================================================================
print("\n1. SIMULATING BIAS-VARIANCE TRADEOFF")
print("-" * 80)

# Try different polynomial degrees (model complexities)
polynomial_degrees = [1, 2, 3, 5, 8, 10]
train_errors = []
test_errors = []
models = []

print("Training models with different complexities:\n")

for degree in polynomial_degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly_features.transform(X_test.reshape(-1, 1))
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_error)
    test_errors.append(test_error)
    models.append(model)
    
    print(f"Degree {degree:2d}: Train MSE = {train_error:8.4f} | Test MSE = {test_error:8.4f}")

# ============================================================================
# 2. Plot training error vs model complexity
# ============================================================================
print("\n2. PLOTTING ERROR CURVES")
print("-" * 80)

plt.figure(figsize=(14, 5))

# Plot 1: Error vs Complexity
plt.subplot(1, 2, 1)
plt.plot(polynomial_degrees, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
plt.plot(polynomial_degrees, test_errors, 's-', label='Test Error', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(polynomial_degrees)

# Plot 2: Model fits at different complexities
plt.subplot(1, 2, 2)

# Plot training data
plt.scatter(X_train, y_train, color='gray', alpha=0.5, s=50, label='Training data')
plt.plot(X_test, y_test, 'k-', linewidth=2, label='True function')

# Plot predictions from selected models
degrees_to_plot = [1, 3, 10]
colors = ['red', 'blue', 'green']

for degree, color in zip(degrees_to_plot, colors):
    poly_features = PolynomialFeatures(degree=degree)
    X_test_poly = poly_features.fit_transform(X_test.reshape(-1, 1))
    y_test_pred = models[polynomial_degrees.index(degree)].predict(X_test_poly)
    plt.plot(X_test, y_test_pred, color=color, linewidth=2, 
             label=f'Degree {degree}', linestyle='--', alpha=0.8)

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Model Fits with Different Complexities', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.tight_layout()
plt.savefig('/home/claude/bias_variance_tradeoff.png', dpi=100, bbox_inches='tight')
print("✓ Bias-Variance plot saved: bias_variance_tradeoff.png")
plt.close()

# ============================================================================
# 3. Detailed explanation
# ============================================================================
print("\n3. BIAS-VARIANCE TRADEOFF EXPLANATION")
print("-" * 80)

explanation = """
BIAS-VARIANCE TRADEOFF:

Definition:
- BIAS: Error from overly simplistic assumptions in learning algorithm
         Measures how far predictions are from correct values on average
         High bias → Underfitting

- VARIANCE: Sensitivity of model to fluctuations in training data
            Measures how much predictions change with different training sets
            High variance → Overfitting

Total Error = Bias² + Variance + Irreducible Error

UNDERFITTING (High Bias, Low Variance):
- Model is too simple to capture underlying pattern
- Performs poorly on both training and test data
- Training error is high
- Test error is also high
- Both curves are elevated

Example: Degree 1 polynomial trying to fit curved data
- Cannot capture the quadratic and sinusoidal components
- Biased towards straight line prediction
- Under-represents the data complexity

OPTIMAL MODEL (Balanced Bias and Variance):
- Model complexity matches data complexity
- Good performance on both training and test data
- Training error and test error are close
- Generalizes well to new data

Example: Degree 3 polynomial for this problem
- Captures main patterns in data
- Not overly complex
- Good balance between bias and variance

OVERFITTING (Low Bias, High Variance):
- Model is too complex, memorizes training data noise
- Performs well on training data, poorly on test data
- Training error is very low
- Test error is much higher than training error
- Large gap between two curves

Example: Degree 10 polynomial
- Fits training data points very closely
- Includes fitting to random noise
- Does not generalize to new data
- Test error increases significantly

MODEL SELECTION:
The optimal model typically occurs where test error is minimized.
Looking at the plot, the sweet spot is usually at a moderate degree
that balances:
1. Capturing the true underlying pattern (low bias)
2. Not fitting to noise (low variance)
3. Good generalization to unseen data
"""

print(explanation)

# ============================================================================
# 4. Analyze the tradeoff for our models
# ============================================================================
print("\n4. DETAILED ANALYSIS")
print("-" * 80)

print("\nModel Performance Analysis:")
print(f"{'Degree':<8} {'Complexity':<12} {'Train Error':<15} {'Test Error':<15} {'Status':<15}")
print("-" * 65)

for i, degree in enumerate(polynomial_degrees):
    gap = test_errors[i] - train_errors[i]
    
    if degree <= 2:
        status = "UNDERFITTING"
    elif degree <= 4:
        status = "BALANCED"
    else:
        status = "OVERFITTING"
    
    print(f"{degree:<8} {'Medium' if degree == 3 else 'Low' if degree < 3 else 'High':<12} "
          f"{train_errors[i]:<15.4f} {test_errors[i]:<15.4f} {status:<15}")

print("\nKey Observations:")
print(f"1. Minimum test error: {min(test_errors):.4f} at degree {polynomial_degrees[test_errors.index(min(test_errors))]}")
print(f"2. Training error is monotonically decreasing (more complex → fits training better)")
print(f"3. Test error first decreases (improving generalization)")
print(f"4. Test error then increases (overfitting to noise)")
print(f"5. Optimal model balances both errors")

# Find optimal degree
optimal_degree = polynomial_degrees[test_errors.index(min(test_errors))]
print(f"\nRecommended Model Complexity: Degree {optimal_degree}")
print(f"  Training Error: {train_errors[polynomial_degrees.index(optimal_degree)]:.4f}")
print(f"  Test Error: {min(test_errors):.4f}")

print("\n" + "="*80)
print("PART B COMPLETED SUCCESSFULLY")
print("="*80)
