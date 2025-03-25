import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, binary_erosion
from skimage.restoration import wiener
# Load a blurred grayscale image
blurred_image = cv2.imread("blurred_image.jpeg", cv2.IMREAD_GRAYSCALE)

# Resize to at least HD resolution
hd_image = cv2.resize(blurred_image, (1920, 1080), interpolation=cv2.INTER_CUBIC)


def deblur_image(blurred, psf, num_iterations=10, lambda_reg=0.01):
    """
    Deblurs an image using an iterative approach with Wiener deconvolution.

    Parameters:
        blurred (numpy.ndarray): The input blurred image.
        psf (numpy.ndarray): The point spread function (PSF).
        num_iterations (int): Number of iterations to refine the deblurred image.
        lambda_reg (float): Regularization parameter to control noise.

    Returns:
        numpy.ndarray: The final deblurred image.
    """
    # Step 1: Convert image to float32 for better precision
    blurred = blurred.astype(np.float32) / 255.0

    # Step 2: Normalize PSF to sum to 1
    psf = psf / np.sum(psf)

    # Step 3: Apply Wiener deconvolution for initial deblurring
    deblurred = wiener(blurred, psf, balance=lambda_reg)

    # Step 4: Apply iterative refinement
    for t in range(num_iterations):
        # Decompose into f_U (smooth) and f_S (sharp details)
        f_U = gaussian_filter(deblurred, sigma=1)  # Smooth component
        f_S = deblurred - f_U  # Detail component

        # Compute Set V using erosion
        binary_mask = f_U > np.mean(f_U)
        V = binary_erosion(binary_mask, structure=psf)

        # Update f_U using only data from V
        f_U_new = np.where(V, blurred, f_U)

        # Non-linear enhancement for f_S
        f_S_new = np.tanh(f_S)

        # Compute unregularized update
        f_unreg = f_U_new + f_S_new

        # Apply final regularization
        deblurred = f_unreg - lambda_reg * convolve2d(f_unreg, psf, mode='same', boundary='symm')

    # Step 5: Normalize and return in 8-bit format
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    return deblurred


# Define a Gaussian PSF kernel
psf_size = 5
psf = cv2.getGaussianKernel(psf_size, 1) * cv2.getGaussianKernel(psf_size, 1).T

# Apply deblurring
hd_image = deblur_image(blurred_image, psf, num_iterations=20)

# Stack both images side-by-side (original on the left, deblurred on the right)
combined_image = np.hstack((blurred_image, hd_image))


# Display the result
cv2.imshow("Deblurred Image", hd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
