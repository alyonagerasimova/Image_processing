import cv2
import numpy as np
import skimage.color
from PIL import Image
from numpy import dstack
import imageio

img = cv2.imread("../img1.jpg")
RGB = np.array(img, dtype=np.uint8)

HSV_skimage = skimage.color.rgb2hsv(RGB.astype(np.uint8))
# cv2.imwrite("HSVLib.png", HSV_skimage.astype(np.uint8))
# Image.fromarray(HSV_skimage.astype(np.uint8)).save("HSVLib.png")
# imageio.imsave('HSVLib.png', HSV_skimage)
# cv2.imshow("hsvLibskimage", HSV_skimage)
# cv2.waitKey(0)


def rgb2hsv(rgb):
    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[:, :, 1] - rgb[:, :, 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[:, :, 2] - rgb[:, :, 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[:, :, 0] - rgb[:, :, 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[:, :, 2] = maxv

    return hsv


HSV = rgb2hsv(RGB)
# cv2.imwrite("HSV.png", HSV)
# imageio.imsave("HSV.png", HSV)
# cv2.imshow("hsv", HSV)
# cv2.waitKey(0)

outArray = np.array(HSV, dtype=np.uint8)
outArrayLib = np.array(HSV_skimage, dtype=np.uint8)
dif = np.abs(outArrayLib - outArray)
cv2.imwrite("difHSV.png", dif)


def hsv2rgb(hsv):
    hi = np.floor(hsv[:, :, 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[:, :, 2].astype('float')
    f = (hsv[:, :, 0] / 60.0) - np.floor(hsv[:, :, 0] / 60.0)
    vmin = v * (1.0 - hsv[:, :, 1])
    vdec = v * (1.0 - (f * hsv[:, :, 1]))
    vinc = v * (1.0 - ((1.0 - f) * hsv[:, :, 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, vinc, vmin))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((vdec, v, vmin))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((vmin, v, vinc))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((vmin, vdec, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((vinc, vmin, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, vmin, vdec))[hi == 5, :]
    return rgb


RGBNew = hsv2rgb(HSV)
# Image.fromarray(RGBNew.astype(np.uint8)).save("RGBLib.png")
# cv2.imwrite("RGBNew.png", RGBNew)
# imageio.imsave("RGBNew.png", RGBNew)
# cv2.imshow("RGBNew", RGBNew)
# cv2.waitKey(0)

RGBNew_skimage = skimage.color.hsv2rgb(HSV_skimage)
cv2.imshow("RGBNewLibskimage", RGBNew_skimage)
cv2.waitKey(0)

outArray = np.array(RGBNew, dtype=np.uint8)
outArrayLib = np.array(RGBNew_skimage, dtype=np.uint8)
dif = np.abs(outArrayLib - outArray)
cv2.imwrite("RGBNew.png", outArray)

# original = cv2.imread("../img1.jpg")
# outArray = np.array(RGBNew, dtype=np.uint8)
# outArrayLib = np.array(original, dtype=np.uint8)
# dif = np.abs(outArrayLib - outArray)
# cv2.imwrite("difRGB.png", dif)
