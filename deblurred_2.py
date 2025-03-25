import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration
from scipy.signal import convolve2d as conv2

# Load a custom color image (no grayscale conversion)
image_path = "blurred_image.jpeg"  # Replace with your own image file
image = io.imread(image_path)  # Load as RGB (assumes 3 channels)

# Ensure image is in float format [0, 1] for processing
image = image.astype(float) / 255.0

# Define a Point Spread Function (PSF) - A simple uniform blur
psf = np.ones((5, 5)) / 25

# Apply blur to each RGB channel separately
blurred_image = np.zeros_like(image)
for channel in range(image.shape[2]):  # Loop over R, G, B channels
    blurred_image[:, :, channel] = conv2(image[:, :, channel], psf, 'same')

# Add Gaussian noise to each channel
rng = np.random.default_rng()
noisy_blurred_image = np.zeros_like(blurred_image)
for channel in range(blurred_image.shape[2]):
    noise = 0.1 * blurred_image[:, :, channel].std() * rng.standard_normal(blurred_image[:, :, channel].shape)
    noisy_blurred_image[:, :, channel] = blurred_image[:, :, channel] + noise
noisy_blurred_image = np.clip(noisy_blurred_image, 0, 1)  # Keep values in [0, 1]

# Deblur using Unsupervised Wiener Deconvolution for each channel
deblurred_image = np.zeros_like(noisy_blurred_image)
for channel in range(noisy_blurred_image.shape[2]):
    deblurred_image[:, :, channel], _ = restoration.unsupervised_wiener(
        noisy_blurred_image[:, :, channel], psf
    )
deblurred_image = np.clip(deblurred_image, 0, 1)  # Ensure valid range

# Display results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].imshow(noisy_blurred_image)  # Display RGB image directly
ax[0].axis('off')
ax[0].set_title('Noisy & Blurred Image')

ax[1].imshow(deblurred_image)
ax[1].axis('off')
ax[1].set_title('Restored Image (Wiener)')

fig.tight_layout()
plt.show()

# Optional: Save the color deblurred image
io.imsave("deblurred_wiener_color_output.jpeg", (deblurred_image * 255).astype(np.uint8))