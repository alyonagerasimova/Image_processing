import cv2
import numpy as np
import colorcorrect.algorithm as cca

orig_image = cv2.imread("../image.png")
cv2.imwrite("../out/grey_world_balance_lib_image.png", cca.grey_world(orig_image))


def grey_world_balance(img):
    b, g, r = cv2.split(img)
    B_ave, G_ave, R_ave = np.mean(b), np.mean(g), np.mean(r)
    Avg = (B_ave + G_ave + R_ave) / 3
    rw = r * Avg / R_ave
    gw = g * Avg / G_ave
    bw = b * Avg / B_ave
    for i in range(len(bw)):
        for j in range(len(bw[0])):
            bw[i][j] = 255 if bw[i][j] > 255 else bw[i][j]
            gw[i][j] = 255 if gw[i][j] > 255 else gw[i][j]
            rw[i][j] = 255 if rw[i][j] > 255 else rw[i][j]

    corrected_img = np.uint8(np.zeros_like(img))
    corrected_img[:, :, 0] = bw
    corrected_img[:, :, 1] = gw
    corrected_img[:, :, 2] = rw
    return corrected_img


balance_image = grey_world_balance(orig_image)
cv2.imwrite("../out/grey_world_balance_image.png", balance_image)
