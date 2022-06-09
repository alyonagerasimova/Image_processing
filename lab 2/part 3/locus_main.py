import colour
import cv2
import numpy as np
from PIL import Image
from colour.plotting import plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931
from matplotlib import pyplot as plt
from numpy import dstack
from collections import deque

matrix_RGB_to_XYZ = np.array(
    [[0.49000, 0.31000, 0.20000],
     [0.17697, 0.81240, 0.01063],
     [0.00000, 0.01000, 0.99000]]
)


def rgb_to_xyz(r0, g0, b0):
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))


# def locus(img):
#     size = 100
#     xMove = np.round(0.312 * size) / 2
#     yMove = np.round(0.329 * size) / 2
#     loscut = np.uint8(np.zeros_like(img))
#     Allx, Ally = [], []
#     tmpx = np.linspace(img[0], len(img), size)
#     tmpy = np.linspace(img[0][0], len(img[0]), size)
#     x, y = [], []
#     for i in range(len(img)):
#         for j in range(len(img[0])):
#             rgb = np.array(img[i, j, :])
#             xyz = rgb_to_xyz(rgb[0], rgb[1], rgb[2])
#             sum = xyz[:, :, 0] + xyz[:, :, 1] + xyz[:, :, 2]
#             if sum > 0:
#                 nx = xyz[:, :, 0] / sum
#                 ny = xyz[:, :, 1] / sum
#                 nz = xyz[:, :, 2] / sum
#                 x = np.round((1 - ny - nz) * size) + xMove
#                 y = np.round((1 - nx - nz) * size * -1) + size - yMove
#                 Allx = np.append(x, tmpx)
#                 Ally = np.append(y, tmpy)
#     plt.figure()
#     plt.scatter(Allx, Ally)
#     plt.show()
#     # loscut.setRGB(x, y, rgb)


# def getColors(imageName):
#     lists = deque()
#     image = Image.open(imageName)
#     pix1 = image.load()
#
#     for i in range(image.width):
#         for j in range(image.height):
#             r = pix1[i, j][0]
#             g = pix1[i, j][1]
#             b = pix1[i, j][2]
#             r = r / 255
#             g = g / 255
#             b = b / 255
#             lists.append([r, g, b])
#
#     return np.array(lists)
#
#
# arr = getColors('../img1.jpg')
# RGB = colour.models.eotf_inverse_sRGB(arr)

# xyz = rgb_to_xyz(img[:, :, 0], img[:, :, 1], img[:, :, 2])
# print(xyz)
# locus(img)

img = cv2.imread("../img1.jpg")
RGB = np.array(img, dtype=np.uint8)

plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(RGB)
