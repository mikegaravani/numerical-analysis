import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, lu_factor, lu_solve, cholesky
from numpy.linalg import cond, norm

def solve_system(n, matrix_type='random'):
    if matrix_type == 'random':
        A = np.random.randn(n, n)  # Random matrix
    elif matrix_type == 'hilbert':
        A = hilbert(n)  # Hilbert matrix

    # Generate a random exact solution vector x_exact
    x_exact = np.random.randn(n)
    b = A @ x_exact

    # Compute condition number of the matrix A
    condition_number = cond(A)

    # Solve the system Ax = b
    try:
        # If A is symmetric and positive definite, use Cholesky
        if np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0):
            L = cholesky(A)
            x_approx = np.linalg.solve(L.T, np.linalg.solve(L, b))
        else:
            raise np.linalg.LinAlgError
    # Else use LU factorization
    except np.linalg.LinAlgError:
        lu, piv = lu_factor(A)
        x_approx = lu_solve((lu, piv), b)

    # Calculate relative error
    relative_error = norm(x_approx - x_exact) / norm(x_exact)

    return condition_number, relative_error

n_values = range(2, 16)
matrix_types = ['random', 'hilbert']

results = {'random': {'n': [], 'condition': [], 'error': []},
           'hilbert': {'n': [], 'condition': [], 'error': []}}

for matrix_type in matrix_types:
    for n in n_values:
        cond_num, rel_error = solve_system(n, matrix_type)
        results[matrix_type]['n'].append(n)
        results[matrix_type]['condition'].append(cond_num)
        results[matrix_type]['error'].append(rel_error)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot for random matrices
plt.subplot(1, 2, 1)
plt.plot(results['random']['n'], results['random']['error'], label='Relative Error (Random)', marker='o')
plt.plot(results['random']['n'], results['random']['condition'], label='Condition Number (Random)', marker='x')
plt.xlabel('Matrix Dimension n')
plt.ylabel('Value')
plt.title('Random Matrix: Error and Condition Number')
plt.legend()

# Plot for Hilbert matrices
plt.subplot(1, 2, 2)
plt.plot(results['hilbert']['n'], results['hilbert']['error'], label='Relative Error (Hilbert)', marker='o')
plt.plot(results['hilbert']['n'], results['hilbert']['condition'], label='Condition Number (Hilbert)', marker='x')
plt.xlabel('Matrix Dimension n')
plt.ylabel('Value')
plt.title('Hilbert Matrix: Error and Condition Number')
plt.legend()

plt.tight_layout()
plt.show()
