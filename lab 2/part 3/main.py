from numpy import dstack
from scipy import signal, interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# lst = pd.read_csv("sample3.csv").values
img = cv2.imread("../img1.jpg")

matrix_RGB_to_XYZ = np.array(
    [[0.49000, 0.31000, 0.20000],
     [0.17697, 0.81240, 0.01063],
     [0.00000, 0.01000, 0.99000]]
)


def rgb_to_xyz(image):
    r0, g0, b0 = cv2.split(image)
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))


X = np.array(rgb_to_xyz(img)[0])
Y = np.array(rgb_to_xyz(img)[1])
Z = np.array(rgb_to_xyz(img)[2])

node = 100

array = np.array([[0.4887180, 0.3106803, 0.2006017],
                  [0.1762044, 0.8129847, 0.0108109],
                  [0.0000000, 0.0102048, 0.9897952]])

InvArray = np.linalg.inv(array)

x = X / (X + Y + Z)
y = Y / (X + Y + Z)
z = Z / (X + Y + Z)

xs, ys, zs = 0, 0, 0
tmpx = np.linspace(x[0], x[-1], node)
tmpy = np.linspace(y[0], y[-1], node)

for i in range(len(x)):
    xs += x[i]
    ys += y[i]
    zs += z[i]

XWhite = xs / (xs + ys + zs)
YWhite = ys / (xs + ys + zs)

Allx, Ally, AllX, AllZ = [], [], [], []

x = np.append(x, tmpx)
y = np.append(y, tmpy)

for (v, v2) in zip(x, y):
    tmpx = np.linspace(XWhite, v, node)
    Allx = np.append(Allx, tmpx)
    tmpy = np.linspace(YWhite, v2, node)
    Ally = np.append(Ally, tmpy)


# AllY = Ally + 0.7 * np.exp(-(Allx - XWhite) ** 2 / 4000 - (Ally - YWhite) ** 2 / 4000)
# AllX = Allx / Ally * AllY
# AllZ = (1 - Allx - Ally) / Ally * AllY
AllY = Ally
AllX = Allx
AllZ = 1 - Allx - Ally

XYZ = [[x, y, z] for (x, y, z) in zip(AllX, AllY, AllZ)]
RGB = [np.dot(InvArray, v) for v in XYZ]
for v in RGB:
    if v[0] < 0:
        v[0] = 0
    if v[0] > 1:
        v[0] = 1
    if v[1] < 0:
        v[1] = 0
    if v[1] > 1:
        v[1] = 1
    if v[2] < 0:
        v[2] = 0
    if v[2] > 1:
        v[2] = 1

plt.figure()
plt.scatter(Allx, Ally, c=RGB)
plt.show()

# fig = plt.subplots()
# y = 1 - x - z
# plt.plot(x, y(x))
# plt.show()
