import numpy as np

# Parameters
m = 100  # number of rows
n = 10   # number of columns (m > n)
np.random.seed(0)  # For reproducibility

# Step 1: Generate random matrix A (m x n)
A = np.random.normal(0, 1, (m, n))

# Step 2: Create the true solution vector alpha (n x 1)
alpha_true = np.ones(n)  # e.g., vector of all 1s

# Step 3: Generate noise vector v with small variance in [0.01, 0.1]
noise_variance = np.random.uniform(0.01, 0.1)
v = np.random.normal(0, np.sqrt(noise_variance), m)

# Step 4: Compute right-hand side y
y = A @ alpha_true + v

# Step 5a: Solve the least squares problem using Normal Equations
AtA = A.T @ A
AtY = A.T @ y
alpha_normal_eq = np.linalg.solve(AtA, AtY)

# Step 5b: Solve the least squares problem using SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / S)
alpha_svd = Vt.T @ S_inv @ U.T @ y

# Step 6: Compute residuals
residual_normal_eq = y - A @ alpha_normal_eq
residual_svd = y - A @ alpha_svd

norm_residual_normal_eq = np.linalg.norm(residual_normal_eq)
norm_residual_svd = np.linalg.norm(residual_svd)

# Print results
print("True solution alpha:", alpha_true)
print("Solution using Normal Equations:", alpha_normal_eq)
print("Solution using SVD:", alpha_svd)
print("\nL2 norm of residuals:")
print(f"Normal Equations residual norm: {norm_residual_normal_eq:.4f}")
print(f"SVD residual norm: {norm_residual_svd:.4f}")