from PIL import Image

img = Image.open("../3-superpixels/image.jpg", 'r')

factor = 0.5
w, h = img.size
new_w = int(w * factor)
new_h = int(h * factor)

img.resize((new_w, new_h), Image.BILINEAR).save('./res.jpg')
