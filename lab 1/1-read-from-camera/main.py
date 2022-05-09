import cv2
import numpy as np

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
winName = "image"
cv2.namedWindow(winName)
ret, frame = camera.read()
cv2.imshow(winName, frame)
cv2.imwrite("test.bmp", frame)
cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()
