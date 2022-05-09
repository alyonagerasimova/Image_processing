import numpy as np
from fast_slic import Slic
from PIL import Image

with Image.open("image.jpg") as f:
    image = np.array(f)

slic = Slic(num_components=1600, compactness=10)
assignment = slic.iterate(image)
print(assignment)
print(slic.slic_model.clusters)
