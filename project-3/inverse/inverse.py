import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import diagsvd
from numpy.linalg import norm

# Copiata funzione gravity da examples.py
def gravity(n, example=1, a=0, b=1, d=0.25):
    dt = 1 / n
    ds = (b - a) / n
    t = dt * (np.arange(1, n + 1) - 0.5)
    s = a + ds * (np.arange(1, n + 1) - 0.5)

    T, S = np.meshgrid(t, s)
    A = dt * d * np.ones((n, n)) / (d**2 + (S - T) ** 2) ** (3 / 2)

    nt = round(n / 3)
    nn = round(n * 7 / 8)
    x = np.ones(n)

    if example == 1:
        x = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)
    elif example == 2:
        x[:nt] = (2 / nt) * np.arange(1, nt + 1)
        x[nt:nn] = ((2 * nn - nt) - np.arange(nt + 1, nn + 1)) / (nn - nt)
        x[nn:] = (n - np.arange(nn + 1, n + 1)) / (n - nn)
    elif example == 3:
        x[:nt] = 2 * np.ones(nt)
    else:
        raise ValueError("Illegal value of example")
    return A, x

def add_noise(y, noise_level):
    noise = noise_level * np.random.randn(*y.shape)
    return y + noise

# Picard Condition visualization
def visualize_picard(A, b):
    U, s, Vt = svd(A)
    UT_b = U.T @ b
    plt.plot(range(1, len(s) + 1), s, label='Singular values')
    plt.plot(range(1, len(UT_b) + 1), abs(UT_b), label='|U.T @ b|')
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Value (log scale)')
    plt.legend()
    plt.title("Picard Condition")
    plt.show()

# Naive solution
def naive_solution(A, b):
    x_naive = np.linalg.pinv(A) @ b
    plt.plot(x_naive, label='Naive solution')
    plt.xlabel("Index")
    plt.title("Naive Solution")
    plt.legend()
    plt.show()
    return x_naive

# TSVD solution
def tsvd_solution(A, b, k):
    U, s, Vt = svd(A)
    S_truncated = diagsvd(s[:k], k, k)
    x_tsvd = Vt.T[:, :k] @ np.linalg.inv(S_truncated) @ U[:, :k].T @ b
    plt.plot(x_tsvd, label=f'TSVD Solution (k={k})')
    plt.xlabel("Index")
    plt.title(f"TSVD Regularized Solution")
    plt.legend()
    plt.show()
    return x_tsvd

# Tikhonov Regularization (L2)
def tikhonov_regularization(A, b, lambd):
    U, s, Vt = svd(A)
    S_reg = np.diag(s / (s**2 + lambd**2))
    x_tikhonov = Vt.T @ S_reg @ U.T @ b
    plt.plot(x_tikhonov, label=f'Tikhonov (λ={lambd})')
    plt.xlabel("Index")
    plt.title("Tikhonov Regularized Solution")
    plt.legend()
    plt.show()
    return x_tikhonov

# Main procedure
n = 100  # Discretization points
noise_levels = [0, 0.01, 0.05, 0.1]  # Different noise levels

for noise_level in noise_levels:
    print(f"Noise Level: {noise_level}")
    A, x_true = gravity(n, example=1)
    b = A @ x_true
    b_noisy = add_noise(b, noise_level)

    # Step 1: Picard Condition
    visualize_picard(A, b_noisy)

    # Step 2: Naive solution
    x_naive = naive_solution(A, b_noisy)

    # Step 3: TSVD Regularization (choose k based on Picard)
    k = 10  # Scelto dopo aver osservato il grafico di Picard
    x_tsvd = tsvd_solution(A, b_noisy, k)

    # Step 4: Tikhonov Regularization (choose lambda using trial-and-error)
    lambd = 0.1
    x_tikhonov = tikhonov_regularization(A, b_noisy, lambd)

    plt.plot(x_true, label='True Solution')
    plt.plot(x_naive, label='Naive Solution', linestyle='dashed')
    plt.plot(x_tsvd, label='TSVD Solution')
    plt.plot(x_tikhonov, label='Tikhonov Solution')
    plt.legend()
    plt.title(f"Comparison at Noise Level {noise_level}")
    plt.show()
