"""
Assignment — Week 04 · Day 21 (AM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part B — Stretch Problem (30%)
Topics: Matrix Operations, Linear Equations, Performance Comparison
"""

import numpy as np
import time

print("="*80)
print("PART B: STRETCH PROBLEM (30%)")
print("="*80)

# ============================================================================
# 1. Implement matrix operations using NumPy
# ============================================================================
print("\n1. MATRIX OPERATIONS")
print("-" * 80)

# Matrix A and B
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix multiplication
print("\n1a. MATRIX MULTIPLICATION (A @ B):")
matrix_product = A @ B
print(matrix_product)

# Alternative using np.dot()
matrix_product_dot = np.dot(A, B)
print(f"\nUsing np.dot() - Same result: {np.allclose(matrix_product, matrix_product_dot)}")

# Transpose
print("\n1b. TRANSPOSE:")
A_transpose = A.T
print(f"A.T =\n{A_transpose}")

# Determinant (only for 2D square matrices)
print("\n1c. DETERMINANT:")
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

# Create a 3x3 matrix with non-zero determinant
C = np.array([[4, 7, 2],
              [3, 5, 1],
              [1, 2, 6]])
det_C = np.linalg.det(C)
print(f"\nMatrix C:\n{C}")
print(f"Determinant of C: {det_C}")

# ============================================================================
# 2. Solve a system of linear equations using NumPy
# ============================================================================
print("\n2. SOLVING SYSTEM OF LINEAR EQUATIONS")
print("-" * 80)

# System: 
# 3x + 2y + z = 11
# 2x + 3y + z = 13
# x + y + 3z = 12

# Coefficient matrix
coeff_matrix = np.array([[3, 2, 1],
                         [2, 3, 1],
                         [1, 1, 3]])

# Constants vector
constants = np.array([11, 13, 12])

print("System of Equations:")
print("3x + 2y + z = 11")
print("2x + 3y + z = 13")
print("x + y + 3z = 12")

# Solve using np.linalg.solve()
solution = np.linalg.solve(coeff_matrix, constants)
print(f"\nSolution (x, y, z): {solution}")

# Verify the solution
print("\nVerification:")
verification = coeff_matrix @ solution
print(f"A @ solution = {verification}")
print(f"Expected = {constants}")
print(f"Match: {np.allclose(verification, constants)}")

# ============================================================================
# 3. Compare performance: Python loop vs NumPy
# ============================================================================
print("\n3. PERFORMANCE COMPARISON: PYTHON LOOP vs NUMPY")
print("-" * 80)

# Create a large array
array_size = 1_000_000
large_array = np.arange(array_size)

print(f"Array size: {array_size:,} elements")

# Method 1: Python loop
print("\nMethod 1: Using Python Loop")
start_time = time.time()
sum_loop = 0
for i in large_array:
    sum_loop += i
end_time = time.time()
time_loop = end_time - start_time
print(f"  Sum: {sum_loop}")
print(f"  Time: {time_loop:.6f} seconds")

# Method 2: NumPy vectorization
print("\nMethod 2: Using NumPy (Vectorized)")
start_time = time.time()
sum_numpy = np.sum(large_array)
end_time = time.time()
time_numpy = end_time - start_time
print(f"  Sum: {sum_numpy}")
print(f"  Time: {time_numpy:.6f} seconds")

# Comparison
print("\nComparison:")
print(f"  Speedup: {time_loop / time_numpy:.1f}x faster with NumPy")
print(f"  Time difference: {time_loop - time_numpy:.6f} seconds")

print("\nExplanation:")
print("  - Python loops are interpreted, executed line by line")
print("  - NumPy operations are vectorized and use optimized C code")
print("  - NumPy operates on contiguous memory blocks efficiently")
print("  - NumPy avoids Python's overhead for each iteration")

# Additional performance test with different operations
print("\n" + "-" * 80)
print("Additional Performance Test: Element-wise Multiplication")

array_size_2 = 100_000
arr1 = np.arange(array_size_2)
arr2 = np.arange(array_size_2)

# Python loop
start_time = time.time()
result_loop = [arr1[i] * arr2[i] for i in range(array_size_2)]
end_time = time.time()
time_loop_2 = end_time - start_time

# NumPy vectorization
start_time = time.time()
result_numpy = arr1 * arr2
end_time = time.time()
time_numpy_2 = end_time - start_time

print(f"Array size: {array_size_2:,}")
print(f"Python loop: {time_loop_2:.6f} seconds")
print(f"NumPy: {time_numpy_2:.6f} seconds")
print(f"Speedup: {time_loop_2 / time_numpy_2:.1f}x faster with NumPy")

print("\n" + "="*80)
print("PART B COMPLETED SUCCESSFULLY")
print("="*80)
