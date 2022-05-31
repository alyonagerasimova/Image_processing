
## Обработка изображений на мобильных устройствах

Сохранение изображения из массива :
```
img = Image.open(path, 'r')
imArr = np.asarray(img)
im = Image.fromarray(imArr)
im.save("picture.jpg")
```