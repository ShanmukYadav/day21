"""
Assignment — Week 04 · Day 21 (AM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Part C — Interview Ready (20%)
"""

import numpy as np

print("="*80)
print("PART C: INTERVIEW READY QUESTIONS (20%)")
print("="*80)

# ============================================================================
# Q1: What is NumPy broadcasting? Why is it useful?
# ============================================================================
print("\nQ1: WHAT IS NUMPY BROADCASTING? WHY IS IT USEFUL?")
print("-" * 80)

answer_q1 = """
NumPy Broadcasting:
Broadcasting is a set of rules that NumPy follows when operating on arrays of 
different shapes. It allows NumPy to work with arrays of different dimensions 
without explicitly reshaping them.

Broadcasting Rules:
1. If arrays have different numbers of dimensions, pad the smaller one with ones
   on the left side of its shape.
2. Arrays are compatible for broadcasting if, for each dimension:
   - The sizes are equal, OR
   - One of them is 1
3. If the sizes don't match and neither is 1, an error is raised.

Why is Broadcasting Useful?
1. Memory Efficiency: Avoids creating copies of data
2. Simplifies Code: No need for explicit loops or reshaping
3. Performance: Operations are vectorized and use optimized C code
4. Readability: Code is more concise and easier to understand
5. Flexibility: Allows operations on arrays of different shapes seamlessly

Examples:
- Adding a scalar to an array
- Adding a 1D array to a 2D array
- Multiplying arrays of different dimensions
"""

print(answer_q1)

# Demonstration
print("\nPractical Examples:")
print("-" * 40)

# Example 1: Scalar + Array
print("Example 1: Scalar + 2D Array")
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
scalar = 10
result = arr + scalar
print(f"Array:\n{arr}")
print(f"Scalar: {scalar}")
print(f"Result (arr + scalar):\n{result}")
print("Broadcasting: Scalar is extended to match array shape")

# Example 2: 1D + 2D
print("\nExample 2: 1D Array + 2D Array")
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
result = arr_2d + arr_1d
print(f"1D Array: {arr_1d}")
print(f"2D Array:\n{arr_2d}")
print(f"Result (2D + 1D):\n{result}")
print("Broadcasting: 1D array is extended to match each row of 2D array")

# ============================================================================
# Q2 (Coding): Implement normalize(X)
# ============================================================================
print("\n\nQ2 (CODING): IMPLEMENT normalize(X)")
print("-" * 80)

def normalize(X):
    """
    Normalize array X to scale values between 0 and 1 using min-max scaling.
    
    Formula: X_normalized = (X - X_min) / (X_max - X_min)
    
    Args:
        X: NumPy array to normalize
        
    Returns:
        Normalized array with values between 0 and 1
    """
    min_val = np.min(X)
    max_val = np.max(X)
    
    if min_val == max_val:
        return np.zeros_like(X, dtype=float)
    
    return (X - min_val) / (max_val - min_val)

print("Function Definition:")
print("""
def normalize(X):
    '''Scale values between 0 and 1 using min-max normalization'''
    min_val = np.min(X)
    max_val = np.max(X)
    
    if min_val == max_val:
        return np.zeros_like(X, dtype=float)
    
    return (X - min_val) / (max_val - min_val)
""")

# Test the function
print("\nTest Cases:")
print("-" * 40)

# Test 1: 1D Array
test_1d = np.array([10, 20, 30, 40, 50])
normalized_1d = normalize(test_1d)
print(f"Test 1 - 1D Array:")
print(f"  Original: {test_1d}")
print(f"  Normalized: {normalized_1d}")
print(f"  Min: {normalized_1d.min()}, Max: {normalized_1d.max()}")

# Test 2: 2D Array
test_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
normalized_2d = normalize(test_2d)
print(f"\nTest 2 - 2D Array:")
print(f"  Original:\n{test_2d}")
print(f"  Normalized:\n{normalized_2d}")
print(f"  Min: {normalized_2d.min():.2f}, Max: {normalized_2d.max():.2f}")

# Test 3: Negative and decimal values
test_3 = np.array([-5, -2, 0, 3, 10])
normalized_3 = normalize(test_3)
print(f"\nTest 3 - Mixed Values:")
print(f"  Original: {test_3}")
print(f"  Normalized: {normalized_3}")
print(f"  Min: {normalized_3.min():.2f}, Max: {normalized_3.max():.2f}")

# ============================================================================
# Q3: What is the difference between vectorisation and loops?
# ============================================================================
print("\n\nQ3: DIFFERENCE BETWEEN VECTORIZATION AND LOOPS")
print("-" * 80)

answer_q3 = """
Vectorization vs Loops:

1. LOOPS (Traditional Python):
   - Process one element at a time
   - Each iteration involves Python interpreter overhead
   - Slow for large datasets
   - Easy to understand but inefficient

2. VECTORIZATION (NumPy):
   - Process entire arrays at once
   - Operations are compiled to optimized C code
   - Fast for large datasets
   - More concise code

Why is NumPy Faster?

1. Implementation:
   - Python loops: Interpreted, line-by-line execution
   - NumPy: Pre-compiled C code, batch processing

2. Memory:
   - Loops: Individual element access, cache misses
   - NumPy: Contiguous memory blocks, better cache locality

3. Overhead:
   - Loops: Python overhead per iteration
   - NumPy: Single function call for entire array

4. Optimization:
   - Loops: No parallelization
   - NumPy: Can use SIMD (Single Instruction Multiple Data)

Performance Example:
For an array of 1,000,000 elements:
- Loop: ~0.1 seconds
- NumPy: ~0.0001 seconds
- NumPy is 1000x FASTER!

Example Code Comparison:
"""

print(answer_q3)

print("\nCode Example: Calculate sum of squares")
print("-" * 40)

arr = np.array([1, 2, 3, 4, 5])

# Loop approach
print("Loop Approach:")
print("""
sum_squares = 0
for x in arr:
    sum_squares += x ** 2
""")
sum_loop = 0
for x in arr:
    sum_loop += x ** 2
print(f"Result: {sum_loop}")

# Vectorized approach
print("\nVectorized Approach:")
print("""
sum_squares = np.sum(arr ** 2)
""")
sum_vectorized = np.sum(arr ** 2)
print(f"Result: {sum_vectorized}")

print(f"\nBoth give same result: {sum_loop == sum_vectorized}")

# Performance comparison
print("\n" + "-" * 40)
print("Performance Comparison on large array:")

import time

large_arr = np.arange(1_000_000)

# Loop
start = time.time()
result_loop = 0
for x in large_arr:
    result_loop += x ** 2
loop_time = time.time() - start

# Vectorized
start = time.time()
result_vec = np.sum(large_arr ** 2)
vec_time = time.time() - start

print(f"Loop time: {loop_time:.6f} seconds")
print(f"Vectorized time: {vec_time:.6f} seconds")
print(f"Speedup: {loop_time / vec_time:.0f}x faster with NumPy")

print("\n" + "="*80)
print("PART C COMPLETED SUCCESSFULLY")
print("="*80)
