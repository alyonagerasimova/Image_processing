from PIL import Image
import numpy

srcArray = numpy.array(Image.open("img.jpg"), dtype=numpy.uint8)
w, h, _ = srcArray.shape
resArray = numpy.zeros((2 * w, 2 * h, 3), dtype=numpy.uint8)
resArray[::2, ::2, 2] = srcArray[:, :, 2]
resArray[1::2, ::2, 1] = srcArray[:, :, 1]
resArray[::2, 1::2, 1] = srcArray[:, :, 1]
resArray[1::2, 1::2, 0] = srcArray[:, :, 0]

Image.fromarray(resArray, "RGB").save("o.png")
