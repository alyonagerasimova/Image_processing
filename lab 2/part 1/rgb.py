import cv2
from numpy import dstack
import numpy as np

img = cv2.imread("../img.jpg")
b, g, r = cv2.split(img)

img_gs = cv2.imread('../img.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imwrite("img2.jpg", img_gs)

cv2.imwrite("r.jpg", r)
cv2.imwrite("b.jpg", b)
cv2.imwrite("g.jpg", g)

linImage = r * 0.2126 + g * 0.7152 + b * 0.0722

img_col = dstack((b, g, r))
merged_img = cv2.merge((b, g, r))


# cv2.imwrite("out.jpg", merged_img)
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


img0 = gamma_trans(img, 1 / 2.2)
cv2.imwrite("out.jpg", img0)

srcArray = np.array(img, dtype=np.uint8)
outArray = np.array(img0, dtype=np.uint8)

print(srcArray - outArray)
cv2.imwrite("dif.jpg", srcArray - outArray)

# def image_gray(image):
#     imageWidth = image.size[0]
#     imageHeight = image.size[1]
#     grayImage = np.array(0, 0)
#     for i in range(imageHeight):
#         for j in range(imageWidth):
#             grayImage[i][j] = int(image[i][j][0] * 0.2126 + image[i][j][1] * 0.7152 + image[i][j][2] * 0.0722)
#     return cv2.LUT(img, grayImage)

# cv2.imwrite("out.jpg", linImage)



# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
#
#
# img = mpimg.imread('image.png')
# gray = rgb2gray(img)
# plt.imshow(gray, cmap=plt.get_cmap('gray'))
# plt.show()


# fi = img_gs / 255.0
# gamma = 0.4
# out = np.power(fi, gamma)
# cv2.imshow("out", out)
# cv2.waitKey()
