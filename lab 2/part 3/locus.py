import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import matrix
import skimage.color
from numpy import dstack
from PIL import Image

img = cv2.imread("../img.jpg")
RGB = np.array(img, dtype=np.uint8)

matrix_RGB_to_XYZ = np.array(
    [[0.412453, 0.357580, 0.180423],
     [0.212671, 0.715160, 0.072169],
     [0.019334, 0.119193, 0.950227]]
)


def rgb_to_xyz(image):
    r0, g0, b0 = cv2.split(image)
    r = 1 / 2.2 * r0 / 255.0
    g = 1 / 2.2 * g0 / 255.0
    b = 1 / 2.2 * b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return xyz_x, xyz_y, xyz_z


X, Y, Z = np.array(rgb_to_xyz(img))


def invert_matrix(AM, IM):
    for fd in range(len(AM)):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(len(AM)):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        for i in list(range(len(AM)))[0:fd] + list(range(len(AM)))[fd + 1:]:
            crScaler = AM[i][fd]
            for j in range(len(AM)):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
    return IM


sum = X + Y + Z
fig = plt.subplots()
x = np.linspace(0, 0.8, 1080)
Z = np.array(Z, dtype=np.uint8)
sum = np.array(sum, dtype=np.uint8)
print(Z.size, sum.size)

y = 1 - x - np.dot(Z, np.linalg.inv(sum))
plt.plot(x, y(x))
plt.show()
