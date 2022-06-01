import cv2
from matplotlib import pyplot as plt

bayer = cv2.imread(r"../2-bayer/img.jpg", -1)
fig = plt.figure()
plt.imshow(bayer, cmap='gray')
plt.title('Input Image')
plt.show()
