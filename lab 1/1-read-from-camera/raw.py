import rawpy
import imageio
import numpy as np
from PIL import Image
from rawkit.raw import Raw

# filename = 'IMG_0001.dng'
# raw_image = Raw(filename)
# buffered_image = np.array(raw_image.to_buffer())
# image = Image.frombytes('RGB', (raw_image.metadata.width, raw_image.metadata.height), buffered_image)
# image.save('image.png', format='png')

raw = rawpy.imread('IMG_0001.dng')
rgb = raw.postprocess()
imageio.imsave('default.jpg', rgb)

