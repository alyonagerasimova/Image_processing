import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

numSegments = 300
image = img_as_float(io.imread("../2-bayer/img.jpg"))
segments = slic(image, n_segments=numSegments)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.show()


# with Image.open("image.jpg") as f:
#     image = np.array(f)
#
# slic = Slic(num_components=1600, compactness=10)
# assignment = slic.iterate(image)
# print(assignment)
# print(slic.slic_model.clusters)
