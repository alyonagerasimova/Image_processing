import cv2
import numpy as np
from numpy import dstack

Rw = 141
Gw = 144
Bw = 153

diag_matrix = np.array([[255 / Rw, 0, 0],
                        [0, 255 / Gw, 0],
                        [0, 0, 255 / Bw]])


def correct_color(image):
    b, g, r = cv2.split(image)
    balance_r = diag_matrix[0][0] * r + diag_matrix[0][1] * g + diag_matrix[0][2] * b
    balance_g = diag_matrix[1][0] * r + diag_matrix[1][1] * g + diag_matrix[1][2] * b
    balance_b = diag_matrix[2][0] * r + diag_matrix[2][1] * g + diag_matrix[2][2] * b
    return dstack((balance_b, balance_g, balance_r))


orig_image = cv2.imread("../image.png")
balance_image = correct_color(orig_image)
cv2.imwrite("../out/balance_image.png", balance_image)
