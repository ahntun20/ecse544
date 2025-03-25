import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve2d as conv2
from pathlib import Path

# Ensure image exists
image_path = "blurred_image.jpeg"  # Replace with your image file
if not Path(image_path).exists():
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Load and preprocess the image
print("Loading image...")
image = io.imread(image_path).astype(float) / 255.0  # RGB, [0, 1]
print(f"Image shape: {image.shape}")

# Define a PSF (e.g., 5x5 uniform blur)
psf = np.ones((5, 5)) / 25

# Simulate blur (replace with real blurry image if available)
print("Simulating blur...")
blurred_image = np.zeros_like(image)
for channel in range(image.shape[2]):
    blurred_image[:, :, channel] = conv2(image[:, :, channel], psf, 'same')

# Add noise
rng = np.random.default_rng()
print("Adding noise...")
noisy_blurred_image = blurred_image + 0.01 * rng.standard_normal(blurred_image.shape)
noisy_blurred_image = np.clip(noisy_blurred_image, 0, 1)

# Tikhonov regularization deconvolution
def tikhonov_deconvolution(blurred_channel, psf, alpha=0.01):
    """
    Deblur a single channel using Tikhonov regularization in the frequency domain.
    :param blurred_channel: Input blurry image channel
    :param psf: Point Spread Function
    :param alpha: Regularization parameter (higher = smoother)
    :return: Deblurred channel
    """
    # Pad PSF to match image size
    psf_padded = np.zeros_like(blurred_channel)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf

    # Fourier transforms
    blurred_fft = np.fft.fft2(blurred_channel)
    psf_fft = np.fft.fft2(psf_padded)

    # Tikhonov formula: H*Y / (|H|^2 + alpha)
    psf_conj = np.conj(psf_fft)  # Complex conjugate of PSF
    denominator = np.abs(psf_fft)**2 + alpha  # Regularization term
    deblurred_fft = blurred_fft * psf_conj / denominator

    # Inverse FFT and clip to valid range
    deblurred_channel = np.real(np.fft.ifft2(deblurred_fft))
    return np.clip(deblurred_channel, 0, 1)

# Apply to each channel
print("Deblurring with Tikhonov regularization...")
deblurred_image = np.zeros_like(noisy_blurred_image)
for channel in range(noisy_blurred_image.shape[2]):
    print(f"Processing channel {channel+1}/{noisy_blurred_image.shape[2]}...")
    deblurred_image[:, :, channel] = tikhonov_deconvolution(
        noisy_blurred_image[:, :, channel], psf, alpha=0.01
    )

# Save result
print("Saving output...")
io.imsave("deblurred_tikhonov_output.jpeg", (deblurred_image * 255).astype(np.uint8))

# Display results
print("Displaying results...")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(noisy_blurred_image)
plt.title("Noisy Blurred Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(deblurred_image)
plt.title("Deblurred (Tikhonov Regularization)")
plt.axis("off")
plt.tight_layout()
plt.show()
print("Done!")