import cv2
import numpy as np
import skimage.color
from numpy import dstack
from PIL import Image

img = cv2.imread("../part 2/img_1.png")

matrix_XYZ_to_RGB = np.array(
    [[2.36461385, -0.89654057, -0.46807328],
     [-0.51516621, 1.4264081, 0.0887581],
     [0.0052037, -0.01440816, 1.00920446]]
)


def xyz_to_rgb(image):
    r0, g0, b0 = cv2.split(image)
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_XYZ_to_RGB[0][0] * r + matrix_XYZ_to_RGB[0][1] * g + matrix_XYZ_to_RGB[0][2] * b
    xyz_y = matrix_XYZ_to_RGB[1][0] * r + matrix_XYZ_to_RGB[1][1] * g + matrix_XYZ_to_RGB[1][2] * b
    xyz_z = matrix_XYZ_to_RGB[2][0] * r + matrix_XYZ_to_RGB[2][1] * g + matrix_XYZ_to_RGB[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))


RGB = xyz_to_rgb(img)
# cv2.imwrite("newRGB.jpg", RGB)
cv2.imshow("newRGB", RGB)
cv2.waitKey(0)

# перевод с использованием библиотечной функции skimage
img = cv2.imread("../part 2/img.png")
# RGB_cv2 = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
RGB_skimage = skimage.color.xyz2rgb(img.astype(np.uint8))
cv2.imshow("newRGBLib", RGB_skimage)
cv2.waitKey(0)

# def gamma_trans(img, gamma):
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
#     return cv2.LUT(img, gamma_table)
#
#
# newImg = gamma_trans(RGB, 2.2)
# cv2.imwrite("newRGBWithGamma.jpg", RGB)
# cv2.imshow("newImg", newImg)
# cv2.waitKey(0)

# original = cv2.imread("../img1.jpg")
# RGB = cv2.imread("../part 2/img_3.png")
# RGB_skimage = cv2.imread("../part 2/img_7.png")
#
#
# srcArray = np.array(original, dtype=np.uint8)
outArray = np.array(RGB, dtype=np.uint8)
outArrayLib = np.array(RGB_skimage, dtype=np.uint8)

dif = np.abs(outArrayLib - outArray)
# dif2 = np.abs(srcArray - outArrayLib)
# cv2.imwrite("difRGBMatrix.jpg", dif)
# cv2.imwrite("difRGBLib.jpg", dif2)
cv2.imwrite("difRGBLib.jpg", dif)
