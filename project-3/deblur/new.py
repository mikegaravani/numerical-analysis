import numpy as np
import matplotlib.pyplot as plt
from skimage import data, restoration
from skimage.util import random_noise
from scipy.signal import convolve2d

# Function to create test problem with PSF
def create_test_problem(img, psf_sigma=1.0):
    x = np.linspace(-3 * psf_sigma, 3 * psf_sigma, 15)
    psf = np.exp(-(x**2 / (2 * psf_sigma**2)))
    psf = np.outer(psf, psf)
    psf /= np.sum(psf)
    blurred_img = convolve2d(img, psf, mode='same', boundary='symm')
    return blurred_img, psf

# Function to add noise
def add_noise(image, noise_level):
    noisy_image = random_noise(image, mode='gaussian', var=noise_level)
    return noisy_image

# Function to compute the L2 error
def compute_error(ground_truth, reconstruction):
    return np.linalg.norm(ground_truth - reconstruction) / np.linalg.norm(ground_truth)

# Tikhonov regularization (L2)
def tikhonov_regularization_image(blurred_img, psf, lambd):
    return restoration.wiener(blurred_img, psf, lambd)

# Total Variation (TV) regularization
def total_variation_regularization(blurred_img, lambd):
    return restoration.denoise_tv_chambolle(blurred_img, weight=lambd)

# Function to plot errors for different regularization levels
def plot_error_graph(errors, lambdas, method_name):
    plt.plot(lambdas, errors, marker='o')
    plt.xlabel("λ (Regularization Parameter)")
    plt.ylabel("Relative L2 Error")
    plt.title(f"{method_name} Error vs λ")
    plt.grid(True)
    plt.show()

# Main procedure
img = data.coins()  # Grayscale image
img = img / np.max(img)  # Normalize to [0, 1]

psf_sigma_values = [1.0, 2.0]  # PSF sizes
noise_levels = [0.0, 0.005, 0.01]  # Noise levels

for psf_sigma in psf_sigma_values:
    for noise_level in noise_levels:
        print(f"PSF Sigma: {psf_sigma}, Noise Level: {noise_level}")

        blurred_img, psf = create_test_problem(img, psf_sigma=psf_sigma)
        noisy_img = add_noise(blurred_img, noise_level)

        # Plot the ground truth, blurred image, and noisy image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(blurred_img, cmap='gray')
        plt.title(f"Blurred (PSF σ={psf_sigma})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f"Noisy (σ²={noise_level})")
        plt.axis('off')
        plt.show()

        # Regularization parameter grid
        lambdas = np.linspace(0.01, 1.0, 10)

        # Track errors for Tikhonov and TV
        tikh_errors = []
        tv_errors = []

        for lambd in lambdas:
            # Tikhonov Regularization
            tikh_reconstruction = tikhonov_regularization_image(noisy_img, psf, lambd)
            tikh_error = compute_error(img, tikh_reconstruction)
            tikh_errors.append(tikh_error)

            # Total Variation Regularization
            tv_reconstruction = total_variation_regularization(noisy_img, lambd)
            tv_error = compute_error(img, tv_reconstruction)
            tv_errors.append(tv_error)

        # Plot Error Graphs for Tikhonov and TV Regularization
        plot_error_graph(tikh_errors, lambdas, f"Tikhonov (Noise={noise_level}, PSF σ={psf_sigma})")
        plot_error_graph(tv_errors, lambdas, f"Total Variation (Noise={noise_level}, PSF σ={psf_sigma})")

        # Comparison of Final Reconstructed Images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_img, cmap='gray')
        plt.title("Noisy Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(tikh_reconstruction, cmap='gray')
        plt.title(f"Tikhonov (λ={lambdas[np.argmin(tikh_errors)]})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(tv_reconstruction, cmap='gray')
        plt.title(f"TV (λ={lambdas[np.argmin(tv_errors)]})")
        plt.axis('off')
        plt.show()
