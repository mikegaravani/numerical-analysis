import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, solve_triangular, svd
from scipy.sparse.linalg import cg

data = pd.read_csv('data_hw.csv')

if set(['x', 'y']).issubset(data.columns):
    x = data['x']
    y = data['y']

def tikhonov_regularization(X, y, lambda_val, L=None):
    A = X.T @ X + lambda_val * (L if L is not None else np.eye(X.shape[1]))
    b = X.T @ y
    return np.linalg.solve(A, b)

# Grado massimo del polinomio
max_degree = 9
lambda_val = 0.01  # Valore di regolarizzazione lambda

# Creazione della matrice di Vandermonde per un polinomio di grado max_degree
degree = max_degree
X = np.vander(x, N=degree + 1, increasing=True)

# 1. Metodo di Cholesky
start_time = time.time()
A = X.T @ X
b = X.T @ y
L = cholesky(A, lower=True)
alpha_cholesky = solve_triangular(L.T, solve_triangular(L, b, lower=True))
time_cholesky = time.time() - start_time

# 2. Metodo SVD
start_time = time.time()
U, s, Vt = svd(X, full_matrices=False)
alpha_svd = Vt.T @ np.diag(1 / s) @ U.T @ y
time_svd = time.time() - start_time

# 3. Metodo dei Gradienti Coniugati (CGLS)
start_time = time.time()
alpha_cgls, info = cg(A, b, maxiter=1000)
time_cgls = time.time() - start_time

# 4. Metodo di Tikhonov (con matrice identità)
alpha_tikhonov_identity = tikhonov_regularization(X, y, lambda_val)

# 5. Metodo di Tikhonov (con matrice L personalizzata)
L_custom = np.eye(degree + 1)
for i in range(degree):
    L_custom[i, i + 1] = -1
alpha_tikhonov_custom = tikhonov_regularization(X, y, lambda_val, L=L_custom)

# Generazione dei valori predetti
y_pred_cholesky = X @ alpha_cholesky
y_pred_svd = X @ alpha_svd
y_pred_cgls = X @ alpha_cgls
y_pred_tikhonov_identity = X @ alpha_tikhonov_identity
y_pred_tikhonov_custom = X @ alpha_tikhonov_custom

# Visualizzazione dei dati e dei fit polinomiali
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color='blue', label='Dati rumorosi', alpha=0.7)
plt.plot(x, y_pred_cholesky, color='red', label='Fit Cholesky', linewidth=3)
plt.plot(x, y_pred_svd, color='green', label='Fit SVD', linewidth=2)
plt.plot(x, y_pred_cgls, color='orange', label='Fit CGLS', linewidth=2)
plt.plot(x, y_pred_tikhonov_identity, color='purple', label='Fit Tikhonov identità', linewidth=2)
plt.plot(x, y_pred_tikhonov_custom, color='brown', label='Fit Tikhonov matrice L', linewidth=2)
plt.title("Scatter plot dei dati con i fit polinomiali (grado 9)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Stampa dei tempi di esecuzione e coefficienti
print(f"Grado del polinomio: {degree}")
print(f"Tempo metodo di Cholesky: {time_cholesky:.6f} secondi. Coefficienti: {alpha_cholesky}")
print(f"Tempo metodo SVD: {time_svd:.6f} secondi. Coefficienti: {alpha_svd}")
print(f"Tempo metodo CGLS: {time_cgls:.6f} secondi. Coefficienti: {alpha_cgls}")
print(f"Coefficiente metodo Tikhonov identità: {alpha_tikhonov_identity}")
print(f"Coefficiente metodo Tikhonov matrice L: {alpha_tikhonov_custom}")