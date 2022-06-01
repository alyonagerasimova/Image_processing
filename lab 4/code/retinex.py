import colorcorrect.algorithm as cca
import cv2

orig_image = cv2.imread("../image.png")
cv2.imwrite("../out/retinex_lib_image.png", cca.retinex(orig_image))
