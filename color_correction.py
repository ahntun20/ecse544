import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the original image
image = cv2.imread("color.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

# Resize to at least HD resolution
hd_image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)

def gray_world_white_balance(img):
    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])
    
    avg_gray = (avg_r + avg_g + avg_b) / 3
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b
    
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_r, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_b, 0, 255)
    
    return img.astype(np.uint8)

corrected_image = gray_world_white_balance(hd_image)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image)
ax[0].set_title("Before processing")
ax[0].axis("off")

ax[1].imshow(corrected_image)
ax[1].set_title("After processing")
ax[1].axis("off")

plt.show()
