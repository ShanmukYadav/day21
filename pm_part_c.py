"""
Assignment — Week 04 · Day 21 (PM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part C — Interview Ready (20%)
"""

import numpy as np

print("="*80)
print("PART C: INTERVIEW READY QUESTIONS (20%)")
print("="*80)

# ============================================================================
# Q1: Difference between Regression and Classification
# ============================================================================
print("\nQ1: DIFFERENCE BETWEEN REGRESSION AND CLASSIFICATION")
print("-" * 80)

q1_answer = """
REGRESSION vs CLASSIFICATION:

1. OUTPUT TYPE:
   
   Regression:
   - Predicts continuous numerical values
   - Output can be any real number
   - Examples: 25.5, -10.2, 1000.99
   - No limits on output range (within model capability)
   
   Classification:
   - Predicts discrete class labels or categories
   - Output is one of predefined classes
   - Examples: Class 0, Class 1, Cat, Dog, Spam, Not Spam
   - Limited to finite set of classes

2. PROBLEM TYPES:

   Regression Examples:
   ✓ Predicting house prices ($)
   ✓ Weather temperature forecasting (°C)
   ✓ Stock price prediction ($)
   ✓ Sales forecasting (units)
   ✓ Age prediction (years)
   ✓ House value estimation ($)
   
   Classification Examples:
   ✓ Email spam detection (Spam / Not Spam)
   ✓ Disease diagnosis (Disease / No Disease)
   ✓ Image recognition (Cat / Dog / Bird)
   ✓ Sentiment analysis (Positive / Negative / Neutral)
   ✓ Credit approval (Approve / Reject)
   ✓ Customer churn prediction (Churn / Stay)

3. EVALUATION METRICS:

   Regression Metrics:
   - Mean Squared Error (MSE): Measures average squared error
     MSE = Σ(y_actual - y_predicted)² / n
   - Root Mean Squared Error (RMSE): Square root of MSE
   - Mean Absolute Error (MAE): Average absolute error
   - R² Score: Proportion of variance explained (0-1)
   
   Classification Metrics:
   - Accuracy: Percentage of correct predictions
     Accuracy = Correct / Total
   - Precision: Of predicted positive, how many are correct
   - Recall: Of actual positive, how many were found
   - F1-Score: Harmonic mean of precision and recall
   - Confusion Matrix: Shows true positives, false positives, etc.

4. MATHEMATICAL MODELS:

   Regression:
   - Linear Regression: y = mx + b
   - Polynomial Regression: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
   - Support Vector Regression (SVR)
   - Neural Networks (continuous output)
   
   Classification:
   - Logistic Regression: P(y=1) = 1 / (1 + e^(-z))
   - Decision Trees
   - Random Forest
   - Support Vector Machines (SVM)
   - Neural Networks (softmax for multi-class)

5. REAL-WORLD EXAMPLES IN DETAIL:

   Example 1: REGRESSION
   Problem: "Predict house prices in a city"
   Input: House features (size, location, age, rooms)
   Output: Price in dollars (e.g., $450,000)
   Model: Linear or Polynomial Regression
   Evaluation: RMSE, R² Score
   
   Example 2: CLASSIFICATION
   Problem: "Identify if a loan should be approved"
   Input: Customer features (income, credit score, debt)
   Output: Approved (1) or Rejected (0)
   Model: Logistic Regression
   Evaluation: Accuracy, Precision, Recall
   
   Example 3: REGRESSION
   Problem: "Forecast daily electricity consumption"
   Input: Temperature, day of week, hour
   Output: Consumption in kilowatts (e.g., 2500.5 kW)
   Model: Neural Network
   Evaluation: MAE, RMSE
   
   Example 4: CLASSIFICATION
   Problem: "Detect fraudulent credit card transactions"
   Input: Transaction amount, merchant, location
   Output: Fraud (1) or Legitimate (0)
   Model: Random Forest
   Evaluation: Precision, Recall, F1-Score

6. KEY DIFFERENCES SUMMARY:

   ASPECT              REGRESSION         CLASSIFICATION
   ─────────────────────────────────────────────────────
   Target Variable     Continuous         Discrete/Categorical
   Possible Outputs    Infinite           Finite
   Prediction Nature   "How much/How many" "What category/Class"
   Evaluation         Error-based metrics  Accuracy-based metrics
   Use Case           Predict values      Predict categories
   Error Type         Large numbers       Wrong classes
   Threshold Needed   No                 Yes (decision boundary)
"""

print(q1_answer)

# ============================================================================
# Q2: Coding - MSE Implementation
# ============================================================================
print("\nQ2 (CODING): IMPLEMENT calculate_mse(y_true, y_pred)")
print("-" * 80)

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error between true and predicted values.
    
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    
    Args:
        y_true: Array of true values (ground truth)
        y_pred: Array of predicted values
        
    Returns:
        Mean Squared Error (float)
        
    Raises:
        ValueError: If arrays have different lengths
    """
    # Convert to numpy arrays for robustness
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check if arrays have same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2
    
    # Return mean of squared errors
    return np.mean(squared_errors)

print("Function Implementation:")
print("""
def calculate_mse(y_true, y_pred):
    '''Calculate Mean Squared Error between true and predicted values'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch")
    
    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)
""")

# Test cases
print("\nTest Cases:")
print("-" * 40)

# Test 1: Perfect predictions
y_true_1 = np.array([1, 2, 3, 4, 5])
y_pred_1 = np.array([1, 2, 3, 4, 5])
mse_1 = calculate_mse(y_true_1, y_pred_1)
print(f"\nTest 1: Perfect Predictions")
print(f"  y_true: {y_true_1}")
print(f"  y_pred: {y_pred_1}")
print(f"  MSE: {mse_1:.4f} (should be 0)")

# Test 2: Off-by-one predictions
y_true_2 = np.array([1, 2, 3, 4, 5])
y_pred_2 = np.array([2, 3, 4, 5, 6])
mse_2 = calculate_mse(y_true_2, y_pred_2)
print(f"\nTest 2: Off-by-One Predictions")
print(f"  y_true: {y_true_2}")
print(f"  y_pred: {y_pred_2}")
print(f"  MSE: {mse_2:.4f} (should be 1)")

# Test 3: Random predictions
y_true_3 = np.array([10, 20, 30, 40, 50])
y_pred_3 = np.array([12, 18, 35, 38, 52])
mse_3 = calculate_mse(y_true_3, y_pred_3)
errors_3 = y_true_3 - y_pred_3
squared_errors_3 = errors_3 ** 2
manual_mse_3 = np.mean(squared_errors_3)
print(f"\nTest 3: Random Predictions")
print(f"  y_true: {y_true_3}")
print(f"  y_pred: {y_pred_3}")
print(f"  Errors: {errors_3}")
print(f"  Squared Errors: {squared_errors_3}")
print(f"  MSE: {mse_3:.4f}")
print(f"  Manual Verification: {manual_mse_3:.4f} ✓")

# Test 4: Error handling
print(f"\nTest 4: Error Handling")
try:
    y_true_4 = np.array([1, 2, 3])
    y_pred_4 = np.array([1, 2])
    mse_4 = calculate_mse(y_true_4, y_pred_4)
except ValueError as e:
    print(f"  Caught error correctly: {e}")

# ============================================================================
# Q3: Bias-Variance Tradeoff Explanation
# ============================================================================
print("\n\nQ3: EXPLAIN BIAS-VARIANCE TRADEOFF")
print("-" * 80)

q3_answer = """
BIAS-VARIANCE TRADEOFF:

DEFINITIONS:

Bias:
- Error due to overly simplistic model assumptions
- How far off are predictions on average from true values
- Result of underfitting
- High bias → Model is too simple for the problem
- Bias doesn't change much with different training datasets
- Systematic error in predictions

Variance:
- Error due to model's sensitivity to training data fluctuations
- How much would predictions change if trained on different data
- Result of overfitting
- High variance → Model is too complex, fits noise
- Variance changes with different training datasets
- Random error in predictions

Total Error = Bias² + Variance + Irreducible Error (noise)

═════════════════════════════════════════════════════════════════

UNDERFITTING (High Bias, Low Variance):

Characteristics:
✗ Model is too simple
✗ Fails to capture underlying patterns
✗ Poor performance on training data
✗ Poor performance on test data
✗ High training error
✗ High test error
✗ Both errors are high and similar

Examples:
- Using degree-1 polynomial to fit curved data
- Linear model for complex non-linear relationship
- Too few features for the problem

Visualization:
    Error
      ↑
      |     ╱
      |    ╱
      |   ╱  
      |  ╱   
      | ╱    
      |_____ Model Complexity →

Visual Pattern:
A straight line trying to fit data with parabolic pattern.

How to Fix:
1. Use more complex model (higher degree polynomial)
2. Add more features to input
3. Reduce regularization
4. Train longer
5. Use better algorithm

═════════════════════════════════════════════════════════════════

OVERFITTING (Low Bias, High Variance):

Characteristics:
✗ Model is too complex
✗ Fits noise and random patterns in training data
✗ Excellent performance on training data
✗ Poor performance on test data
✗ Large gap between training and test error
✓ Low training error
✗ High test error
✗ Cannot generalize to new data

Examples:
- Using degree-20 polynomial to fit degree-3 data
- Decision tree that creates separate leaf for each training sample
- Neural network with too many parameters
- Features engineered with noise

Visualization:
    Error
      ↑  Test Error
      |        ╱
      |       ╱
      |      ╱
      |     ╱
      |    ╱ Training Error
      |   ╱
      |  ╱___
      |_________ Model Complexity →

Visual Pattern:
Wiggly curve fitting through every data point, including noise.

How to Fix:
1. Use simpler model (lower degree polynomial)
2. Add regularization (L1, L2, Dropout)
3. Use more training data
4. Use cross-validation
5. Early stopping
6. Feature selection (remove noise features)

═════════════════════════════════════════════════════════════════

OPTIMAL MODEL (Balanced Bias and Variance):

Characteristics:
✓ Model complexity matches data complexity
✓ Good training error
✓ Good test error
✓ Small gap between training and test error
✓ Generalizes well to unseen data
✓ Captures underlying pattern without fitting noise

Visualization:
    Error
      ↑  
      |   
      |    Test Error
      |    ╱╲
      |   ╱  ╲
      |  ╱    ╲  Training Error
      | ╱______╲___
      |____________ Model Complexity →
                 ↑
              Optimal
              Point

Trade-off:
- As complexity increases, bias decreases
- As complexity increases, variance increases
- Total error is U-shaped
- Optimal point is at the minimum of the U-curve

═════════════════════════════════════════════════════════════════

PRACTICAL EXAMPLE: House Price Prediction

Degree 1 (Underfitting):
- Uses simple linear model
- Assumes price = a*size + b
- Ignores location, condition, age
- Poor predictions for both training and test
- Bias is high, Variance is low

Degree 3 (Balanced):
- Uses cubic polynomial
- Captures main price-size relationship
- Ignores noise in data
- Good predictions for both training and test
- Bias is moderate, Variance is moderate

Degree 15 (Overfitting):
- Uses very complex polynomial
- Fits every quirk in training data
- Predicts well on training data
- Predicts poorly on new houses
- Bias is low, Variance is high

═════════════════════════════════════════════════════════════════

SOLUTIONS TO BIAS-VARIANCE PROBLEMS:

For Underfitting (High Bias):
1. Increase model complexity
2. Reduce regularization parameter (λ)
3. Train longer / More epochs
4. Engineer better features
5. Use non-linear models
6. Increase model capacity

For Overfitting (High Variance):
1. Decrease model complexity
2. Add regularization (L1/L2, Ridge, Lasso)
3. Use dropout in neural networks
4. Collect more training data
5. Remove irrelevant features
6. Use ensemble methods
7. Early stopping

═════════════════════════════════════════════════════════════════

KEY INSIGHT:
The goal is to find the sweet spot where total error is minimized.
This is neither too simple (underfitting) nor too complex (overfitting).
It's where the model generalizes best to unseen data.
"""

print(q3_answer)

print("\n" + "="*80)
print("PART C COMPLETED SUCCESSFULLY")
print("="*80)
