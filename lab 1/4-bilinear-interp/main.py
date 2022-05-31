from PIL import Image
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

factor = 0.5
path = "../2-bayer/img.jpg"


# def BilinearInterpolation(imArr, pix, posX, posY):
#     X_int = int(posX)
#     Y_int = int(posY)
#     X_float = posX - X_int
#     Y_float = posY - Y_int
#     X_int_inc = min(X_int + 1, imArr.shape[1] - 1)
#     Y_int_inc = min(Y_int + 1, imArr.shape[0] - 1)
#
#     bl = pix[Y_int, X_int]
#     br = pix[Y_int, X_int_inc]
#     tl = pix[Y_int_inc, X_int]
#     tr = pix[Y_int_inc, X_int_inc]
#
#     b = X_float * br + (1. - X_float) * bl
#     t = X_float * tr + (1. - X_float) * tl
#     result = int(Y_float * t + (1. - Y_float) * b + 0.5)
#
#     return result
#
#
# img = Image.open(path, 'r')
# imArr = np.asarray(img)
# pix = img.load()
# k = factor
# newShape = list(map(int, [imArr.shape[0] * k, imArr.shape[1] * k]))
# resultImg = np.empty(newShape, dtype=np.uint8)
# rowScale = float(imArr.shape[0]) / float(resultImg.shape[0])
# colScale = float(imArr.shape[1]) / float(resultImg.shape[1])
#
# for r in range(resultImg.shape[0]):
#     for c in range(resultImg.shape[1]):
#         old_r = r * rowScale
#         old_c = c * colScale
#         resultImg[c, r] = BilinearInterpolation(imArr, pix, old_c,
#                                                 old_r)
#
# plt.imshow(np.uint8(resultImg))
# plt.show()

########################################################################################################################

img = Image.open(path, 'r')

w, h = img.size
new_w = int(w * factor)
new_h = int(h * factor)

img.resize((new_w, new_h), Image.BILINEAR).save('./res.jpg')

