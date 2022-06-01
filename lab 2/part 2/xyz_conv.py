import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from numpy import dstack

img = cv2.imread("../img1.jpg")
RGB = np.array(img, dtype=np.uint8)

# img = plt.imread("../img1.jpg")
# plt.imshow(np.array(RGB))
# plt.title('Input Image')
# plt.show()

matrix_RGB_to_XYZ = np.array(
    [[0.49000, 0.31000, 0.20000],
     [0.17697, 0.81240, 0.01063],
     [0.00000, 0.01000, 0.99000]]
)

# перевод с использованием библиотечной функции
XYZ_skimage = skimage.color.rgb2xyz(RGB.astype(np.uint8))
# XYZ_cv2 = cv2.cvtColor(RGB, cv2.COLOR_RGB2XYZ)
cv2.imwrite("xyzLibskimage.png", np.array(XYZ_skimage))
cv2.imshow("xyzLibskimage", XYZ_skimage)
cv2.waitKey(0)
# plt.imshow(np.array(XYZ_skimage))
# plt.title('Input Image')
# plt.show()


# перевод с использованием матрицы
def rgb_to_xyz(image):
    r0 = image[:, :, 0]
    g0 = image[:, :, 1]
    b0 = image[:, :, 2]
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))


XYZ = rgb_to_xyz(img)
cv2.imwrite("xyz.png", np.array(XYZ))
# plt.imshow(np.array(XYZ))
# plt.title('Input Image')
# plt.show()
cv2.imshow("xyz", XYZ)
cv2.waitKey(0)

srcArray = np.array(XYZ_skimage, dtype=np.uint8)
outArray = np.array(XYZ, dtype=np.uint8)

r0 = srcArray[:, :, 0]
g0 = srcArray[:, :, 1]
b0 = srcArray[:, :, 2]
r = outArray[:, :, 0]
g = outArray[:, :, 1]
b = outArray[:, :, 2]

dif_r = np.abs(r0 - r)
dif_g = np.abs(g0 - g)
dif_b = np.abs(b0 - b)
dif = np.abs(srcArray - outArray)
print(dif)
cv2.imwrite("dif.jpg", dif)
# plt.imshow(np.array(dif))
# plt.title('Input Image')
# plt.show()
# cv2.imshow("dif_r", dif_r)
# cv2.imshow("dif_g", dif_g)
# cv2.imshow("dif_b", dif_b)
# cv2.waitKey(0)
