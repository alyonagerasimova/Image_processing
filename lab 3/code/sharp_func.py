from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2

#ImageEnhance.Sharpness()
image = Image.open("../out/blur.jpg")
factor = 15
enhancer_object = ImageEnhance.Sharpness(image)
out = enhancer_object.enhance(factor)
out.save("out/sharpness.jpg")

#Ядро и функция filter2D
image = cv2.imread('../out/blur.jpg')
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
im = cv2.filter2D(image, -1, kernel)
cv2.imwrite("../out/sharpness2.jpg", im)
