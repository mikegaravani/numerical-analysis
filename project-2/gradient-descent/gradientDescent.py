import numpy as np
import matplotlib.pyplot as plt

def relative_error(x_true, x_k):
    return np.linalg.norm(x_true - x_k) / np.linalg.norm(x_true)

def gradient_descent_fixed_alpha_tracking(f, grad_f, alpha, maxit=1000, tolf=1e-6, tolx=1e-6, n=2, x_true=None):
    x = np.zeros(n)
    grad_norms = []
    errors = []
    for i in range(maxit):
        grad = grad_f(x)
        grad_norms.append(np.linalg.norm(grad))

        if x_true is not None:
            errors.append(relative_error(x_true, x))

        x_new = x - alpha * grad
        if np.linalg.norm(f(x_new) - f(x)) < tolf or np.linalg.norm(x_new - x) < tolx:
            break
        x = x_new

    return grad_norms, errors

def gradient_descent_backtracking_tracking(f, grad_f, maxit=1000, tolf=1e-6, tolx=1e-6, alpha=1.0, rho=0.5, c=1e-4, n=2, x_true=None):
    x = np.zeros(n)
    grad_norms = []
    errors = []
    for i in range(maxit):
        grad = grad_f(x)
        grad_norms.append(np.linalg.norm(grad))

        if x_true is not None:
            errors.append(relative_error(x_true, x))

        step_size = alpha
        while f(x - step_size * grad) > f(x) - c * step_size * np.linalg.norm(grad)**2:
            step_size *= rho

        x_new = x - step_size * grad
        if np.linalg.norm(f(x_new) - f(x)) < tolf or np.linalg.norm(x_new - x) < tolx:
            break
        x = x_new

    return grad_norms, errors

# Function 1
def func1(x):
    return (x[0] - 3)**2 + (x[1] - 1)**2

def grad_func1(x):
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 1)])

# Function 2
def func2(x):
    return 10 * (x[0] - 1)**2 + (x[1] - 2)**2

def grad_func2(x):
    return np.array([20 * (x[0] - 1), 2 * (x[1] - 2)])

# Function 3
def func3(x):
    return x[0]**4 + x[0]**3 - 2 * x[0]**2 - 2 * x[0]

def grad_func3(x):
    return np.array([4 * x[0]**3 + 3 * x[0]**2 - 4 * x[0] - 2])


# --- 1. Plot Gradient Norms ---
alpha_values = [0.01, 0.1, 0.5]  # Different alpha values for fixed step size

plt.figure(figsize=(10, 6))

# Function 1 - Gradient Norm Plot
for alpha in alpha_values:
    grad_norms_fixed, _ = gradient_descent_fixed_alpha_tracking(func1, grad_func1, alpha=alpha, n=2, x_true=np.array([3, 1]))
    plt.plot(grad_norms_fixed, label=f"Fixed alpha = {alpha}")

grad_norms_backtracking, _ = gradient_descent_backtracking_tracking(func1, grad_func1, n=2, x_true=np.array([3, 1]))
plt.plot(grad_norms_backtracking, label="Backtracking", linestyle='--')

plt.xlabel("Iterations")
plt.ylabel(r"$||\nabla f(x_k)||_2$")
plt.title("Norma del Gradiente durante le Iterazioni (Funzione 1)")
plt.legend()
plt.show()

# --- 2. Plot Relative Errors ---
plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    _, errors_fixed = gradient_descent_fixed_alpha_tracking(func1, grad_func1, alpha=alpha, n=2, x_true=np.array([3, 1]))
    plt.plot(errors_fixed, label=f"Fixed alpha = {alpha}")

_, errors_backtracking = gradient_descent_backtracking_tracking(func1, grad_func1, n=2, x_true=np.array([3, 1]))
plt.plot(errors_backtracking, label="Backtracking", linestyle='--')

plt.xlabel("Iterations")
plt.ylabel("Relative Error")
plt.title("Errore Relativo rispetto alla Soluzione Esatta (Funzione 1)")
plt.legend()
plt.show()

# --- 3. Plot Function 3 ---
x_vals = np.linspace(-3, 3, 100)
y_vals = [func3([x]) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r"$f(x) = x^4 + x^3 - 2x^2 - 2x$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Grafico della Funzione f(x) nell'intervallo [-3, 3]")
plt.axvline(x=0.9222, color='r', linestyle='--', label="Minimo")
plt.legend()
plt.show()

# --- Run Gradient Descent with Different Initial Points ---
initial_points = [-2, -1, 0, 1, 2]

plt.figure(figsize=(10, 6))
for x0 in initial_points:
    _, errors_backtracking_func3 = gradient_descent_backtracking_tracking(func3, grad_func3, n=1, x_true=np.array([0.9222]))
    plt.plot(errors_backtracking_func3, label=f"Backtracking x0 = {x0}")

plt.xlabel("Iterations")
plt.ylabel("Relative Error")
plt.title("Convergenza del Metodo GD con Backtracking per Vari x0 (Funzione 3)")
plt.legend()
plt.show()
