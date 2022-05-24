import cv2
from PIL import Image, ImageFilter
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# img = plt.imread('../lab 2/img.jpg')
# t = np.linspace(-10, 10, 30)
# bump = np.exp(-0.1*t**2)
# bump /= np.trapz(bump)
# kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
# kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))
# img_ft = fftpack.fft2(img, axes=(0, 1))
# img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
# img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
# img2 = np.clip(img2, 0, 1)
# plt.figure()
# plt.imshow(img2)
# plt.imsave("out/gaussian_blur1.jpg", img2)

#Размытие по Гауссу
OriImage = Image.open('../../lab 2/img.jpg')
gaussImage = OriImage.filter(ImageFilter.GaussianBlur(5))
gaussImage.save('out/gaussian_blur.jpg')

#GaussianBlur( кадр, размер ядра, отклонение)
image = cv2.imread('../../lab 2/img.jpg')
Gaussian = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite("../out/blur.jpg", Gaussian)
# cv2.imshow('Gaussian Blurring', Gaussian)
# cv2.waitKey(0)
