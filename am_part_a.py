"""
Assignment — Week 04 · Day 21 (AM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part A — Concept Application (40%)
Topics: NumPy - Array Operations, Broadcasting, Indexing, Vectorisation
"""

import numpy as np
import pandas as pd

print("="*80)
print("PART A: CONCEPT APPLICATION (40%)")
print("="*80)

# ============================================================================
# 1. Create NumPy arrays of different dimensions (1D, 2D, 3D)
# ============================================================================
print("\n1. CREATING ARRAYS OF DIFFERENT DIMENSIONS")
print("-" * 80)

# 1D Array
array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("1D Array:")
print(f"  Shape: {array_1d.shape}")
print(f"  Data: {array_1d}")

# 2D Array
array_2d = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])
print("\n2D Array (3x4):")
print(f"  Shape: {array_2d.shape}")
print(f"  Data:\n{array_2d}")

# 3D Array
array_3d = np.array([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]])
print("\n3D Array (3x2x2):")
print(f"  Shape: {array_3d.shape}")
print(f"  Data:\n{array_3d}")

# ============================================================================
# 1a. Perform indexing and slicing operations
# ============================================================================
print("\n1a. INDEXING AND SLICING OPERATIONS")
print("-" * 80)

print("1D Array Indexing:")
print(f"  array_1d[0]: {array_1d[0]}")
print(f"  array_1d[-1]: {array_1d[-1]}")
print(f"  array_1d[2:5]: {array_1d[2:5]}")

print("\n2D Array Indexing:")
print(f"  array_2d[0]: {array_2d[0]}")
print(f"  array_2d[0, 0]: {array_2d[0, 0]}")
print(f"  array_2d[1, 2]: {array_2d[1, 2]}")

print("\n2D Array Slicing:")
print(f"  array_2d[0:2, :]: \n{array_2d[0:2, :]}")
print(f"  array_2d[:, 1:3]: \n{array_2d[:, 1:3]}")
print(f"  array_2d[::2, ::2]: \n{array_2d[::2, ::2]}")

print("\n3D Array Indexing:")
print(f"  array_3d[0]: \n{array_3d[0]}")
print(f"  array_3d[1, 0, 1]: {array_3d[1, 0, 1]}")

# ============================================================================
# 1b. Extract specific rows, columns, and subarrays
# ============================================================================
print("\n1b. EXTRACTING ROWS, COLUMNS, AND SUBARRAYS")
print("-" * 80)

print("Extract Row 1:")
row_1 = array_2d[1, :]
print(f"  {row_1}")

print("\nExtract Column 2:")
col_2 = array_2d[:, 2]
print(f"  {col_2}")

print("\nExtract Subarray (rows 0-1, cols 1-2):")
subarray = array_2d[0:2, 1:3]
print(f"  {subarray}")

# ============================================================================
# 2. Implement basic operations using NumPy (without loops)
# ============================================================================
print("\n2. BASIC NUMPY OPERATIONS (VECTORIZED)")
print("-" * 80)

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 10])

print(f"Array A: {a}")
print(f"Array B: {b}")

# Element-wise operations
print("\nElement-wise Operations:")
print(f"  Addition (A + B): {a + b}")
print(f"  Subtraction (A - B): {a - b}")
print(f"  Multiplication (A * B): {a * b}")
print(f"  Division (B / A): {b / a}")

# Statistical operations
print("\nStatistical Operations:")
print(f"  Mean of A: {np.mean(a)}")
print(f"  Variance of A: {np.var(a)}")
print(f"  Std Deviation of A: {np.std(a)}")
print(f"  Mean of 2D array:\n{np.mean(array_2d, axis=0)}")  # Column-wise
print(f"  Mean of 2D array:\n{np.mean(array_2d, axis=1)}")  # Row-wise

# ============================================================================
# 3. Demonstrate broadcasting
# ============================================================================
print("\n3. BROADCASTING")
print("-" * 80)

# Case 1: Add 1D array to 2D array
print("Case 1: Add 1D array to 2D array")
array_2d_small = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
vector = np.array([10, 20, 30])
result_broadcast_1 = array_2d_small + vector
print(f"  2D Array:\n{array_2d_small}")
print(f"  1D Vector: {vector}")
print(f"  Result (Broadcasting):\n{result_broadcast_1}")
print("  Explanation: The 1D array is broadcasted to match the 2D array shape (3x3)")

# Case 2: Multiply matrix by scalar
print("\nCase 2: Multiply matrix by scalar")
scalar = 5
result_broadcast_2 = array_2d_small * scalar
print(f"  Matrix:\n{array_2d_small}")
print(f"  Scalar: {scalar}")
print(f"  Result:\n{result_broadcast_2}")
print("  Explanation: Scalar is broadcasted to all elements")

# Case 3: Multiply matrix by vector
print("\nCase 3: Multiply matrix by column vector")
col_vector = np.array([[2], [3], [4]])  # Shape (3, 1)
result_broadcast_3 = array_2d_small * col_vector
print(f"  Matrix:\n{array_2d_small}")
print(f"  Column Vector:\n{col_vector}")
print(f"  Result:\n{result_broadcast_3}")
print("  Explanation: Column vector is broadcasted across columns")

# ============================================================================
# 4. Implement vectorised operations
# ============================================================================
print("\n4. VECTORIZED OPERATIONS")
print("-" * 80)

test_array = np.array([1, 2, -3, 4, -5, 6, -7, 8])
print(f"Original Array: {test_array}")

# Square of all elements
squared = test_array ** 2
print(f"Squares: {squared}")

# Cube of all elements
cubed = test_array ** 3
print(f"Cubes: {cubed}")

# Replace all negative values with 0
array_positive = np.where(test_array < 0, 0, test_array)
print(f"Replace Negatives with 0: {array_positive}")

# Normalize array (scale between 0 and 1)
array_to_normalize = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
min_val = np.min(array_to_normalize)
max_val = np.max(array_to_normalize)
normalized = (array_to_normalize - min_val) / (max_val - min_val)
print(f"\nOriginal Array: {array_to_normalize}")
print(f"Min: {min_val}, Max: {max_val}")
print(f"Normalized (0-1): {normalized}")

# ============================================================================
# 5. Given a dataset (NumPy array)
# ============================================================================
print("\n5. DATASET OPERATIONS")
print("-" * 80)

# Create a dataset
dataset = np.array([45, 23, 89, 12, 95, 34, 67, 88, 23, 55, 
                    100, 5, 76, 42, 81, 60, 39, 72, 15, 99])
print(f"Dataset: {dataset}")

# Find top 5 maximum values
top_5_max = np.sort(dataset)[-5:][::-1]
print(f"\nTop 5 Maximum Values: {top_5_max}")

# Create a 2D dataset for row-wise and column-wise sums
dataset_2d = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])
print(f"\n2D Dataset:\n{dataset_2d}")

# Row-wise sums
row_sums = np.sum(dataset_2d, axis=1)
print(f"Row-wise Sums: {row_sums}")

# Column-wise sums
col_sums = np.sum(dataset_2d, axis=0)
print(f"Column-wise Sums: {col_sums}")

# Identify indices of values greater than threshold
threshold = 5
indices_above_threshold = np.where(dataset_2d > threshold)
print(f"\nIndices where values > {threshold}:")
print(f"  Row indices: {indices_above_threshold[0]}")
print(f"  Col indices: {indices_above_threshold[1]}")
print(f"  Values: {dataset_2d[indices_above_threshold]}")

print("\n" + "="*80)
print("PART A COMPLETED SUCCESSFULLY")
print("="*80)
