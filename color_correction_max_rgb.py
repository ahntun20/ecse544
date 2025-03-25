import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread("color.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Resize to at least HD resolution
hd_image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)

def max_rgb_white_balance(img):
    """
    Applies Max-RGB White Balance algorithm.
    """
    # Split channels
    R, G, B = cv2.split(img)
    
    # Find the maximum intensity in each channel
    max_R = np.max(R)
    max_G = np.max(G)
    max_B = np.max(B)

    # Normalize each channel using its max value
    R = np.clip((R / max_R) * 255, 0, 255)
    G = np.clip((G / max_G) * 255, 0, 255)
    B = np.clip((B / max_B) * 255, 0, 255)

    # Merge channels back
    corrected_img = cv2.merge((R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8)))
    
    return corrected_img

# Apply Max-RGB color correction
corrected_image = max_rgb_white_balance(hd_image)

# Display results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(hd_image)
ax[0].set_title("Before processing")
ax[0].axis("off")

ax[1].imshow(corrected_image)
ax[1].set_title("After processing")
ax[1].axis("off")

plt.show()
