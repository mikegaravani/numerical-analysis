import numpy as np
import matplotlib.pyplot as plt

# Parameters
m, n = 100, 50 # Matrix with 100 rows and 50 columns
np.random.seed(41)
A = np.random.normal(0, 1, (m, n))

# The real solution is a vector of ones
alpha_true = np.ones(n)

noise_variance = np.random.uniform(0.01, 0.1)
v = np.random.normal(0, np.sqrt(noise_variance), m)

y = A @ alpha_true + v

# 1. Normal Equations
alpha_normal_eq = np.linalg.inv(A.T @ A) @ (A.T @ y)

# 2. SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
alpha_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ (U.T @ y)

residual_normal_eq = np.linalg.norm(A @ alpha_normal_eq - y, 2)
residual_svd = np.linalg.norm(A @ alpha_svd - y, 2)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(alpha_true, label='True Solution (alpha)', marker='o', linestyle='dotted')
plt.plot(alpha_normal_eq, label='Normal Equations Solution', marker='x', linestyle='--')
plt.plot(alpha_svd, label='SVD Solution', marker='s', linestyle=':')
plt.title('Solutions Comparison')
plt.xlabel('Index')
plt.ylabel('Alpha Values')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(['Normal Equations', 'SVD'], [residual_normal_eq, residual_svd], color=['blue', 'green'])
plt.title('Residual Norms Comparison')
plt.ylabel('Residual Norm (L2)')
plt.grid(axis='y')

plt.tight_layout()
plt.show()
