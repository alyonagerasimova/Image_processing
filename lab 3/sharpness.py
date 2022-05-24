import cv2
import numpy as np

def sharp(image, kernel_size=(15, 15), sigma=0.0, amount=5.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


image = cv2.imread('out/blur.jpg')
sharpened_image = sharp(image)
cv2.imwrite('out/my-sharpened-image5.jpg', sharpened_image)
