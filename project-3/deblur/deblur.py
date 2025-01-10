import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

# Load default cameraman image
image = img_as_float(data.camera())
ground_truth = image.copy()

# Apply Gaussian blur
blurred_image = gaussian_filter(image, sigma=3)

# Flatten image to create the right-hand side vector b
b = blurred_image.flatten()

# Define operator A as a LinearOperator
shape = image.shape
A_op = LinearOperator(
    (b.size, b.size),
    matvec=lambda x: gaussian_filter(x.reshape(shape), sigma=3).flatten()
)

# Function to calculate CGLS solution
def cgls(A, b, max_iter=500, rtol=1e-3):
    x0 = np.zeros_like(b)
    x, info = cg(A, b, x0=x0, maxiter=max_iter, atol=0, rtol=rtol)
    if info != 0:
        print(f"Warning: CG did not converge after {max_iter} iterations (info={info}).")
    return x.reshape(shape)

# Naive least squares solution
cgls_solution = cgls(A_op, b)
plt.figure()
plt.title("CGLS Naive Least Squares Solution")
plt.imshow(cgls_solution, cmap='gray')
plt.axis('off')
plt.show()

# Tikhonov Regularization
def tikhonov_regularization(A, b, lambdas):
    errors = []
    solutions = []

    for lam in lambdas:
        I = np.identity(len(b))
        reg_matrix = LinearOperator(
            (b.size, b.size),
            matvec=lambda x: A.matvec(x) + lam * x
        )
        x_reg = cgls(reg_matrix, b)
        solutions.append(x_reg)
        error = np.linalg.norm(x_reg - ground_truth, ord=1)
        errors.append(error)

    best_lambda_idx = np.argmin(errors)
    best_solution = solutions[best_lambda_idx]

    plt.plot(lambdas, errors)
    plt.title("Tikhonov Regularization Error vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("L1 Error")
    plt.show()

    return best_solution

lambdas = np.logspace(-4, 1, 50)
tikhonov_solution = tikhonov_regularization(A_op, b, lambdas)
plt.figure()
plt.title("Tikhonov Regularized Solution")
plt.imshow(tikhonov_solution, cmap='gray')
plt.axis('off')
plt.show()

# Total Variation Regularization (TV)
def total_variation_regularization(image, lambdas):
    errors = []
    solutions = []

    for lam in lambdas:
        tv_solution = denoise_tv_chambolle(blurred_image, weight=lam)
        solutions.append(tv_solution)
        error = np.linalg.norm(tv_solution - ground_truth, ord=2)
        errors.append(error)

    best_lambda_idx = np.argmin(errors)
    best_solution = solutions[best_lambda_idx]

    plt.plot(lambdas, errors)
    plt.title("Total Variation Regularization Error vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("L2 Error")
    plt.show()

    return best_solution

tv_lambdas = np.linspace(0.001, 0.2, 50)
tv_solution = total_variation_regularization(blurred_image, tv_lambdas)
plt.figure()
plt.title("Total Variation Regularized Solution")
plt.imshow(tv_solution, cmap='gray')
plt.axis('off')
plt.show()
