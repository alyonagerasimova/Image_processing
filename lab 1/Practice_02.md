## Лабораторная работа 1. Получение изображений. Работа с RAW изображениями. Дебайеризация. Библиотеки работы с изображениями

1. Подготовка среды программирования
2. Поиск библиотек для работы с изображениями (OpenCV, Scikit-Image, Scipy, Python Image Library (Pillow/PIL), Matplotlib, SimpleITK, Numpy, Mahotas, Сolour)
3. Чтение изображений с камеры устройства

```
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
winName = "image"
cv2.namedWindow(winName)
ret, frame = camera.read()
cv2.imshow(winName, frame)
cv2.imwrite("test.bmp", frame)
cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()
```

<div align="center">
 <div>Изображение:</div>
 <img src="" width=66%>
</div>

4. Получение RAW изображения с устройства.

```

```

<div align="center">
<div>Изображение:</div>
 <img src="" alt="" width=66%>
</div>

5. Создание алгоритма "байеризации".

```
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

```

<div align="center">
<div>"Байеризационное" изображение:</div>
 <img src="2-bayer/o.png" alt="" width=66%>
</div>

6. Выбор изображения для работы

7. Реализация суперпикселей. Аналоги библиотек.

<div align="center">
 <img src="" alt="" width=66%>
</div>

8. Реализация билинейной интерполяции. Аналоги библиотек.

```

```
<div>
 <div>Результат:</div>
 <img src="" alt="sharpness" width=66%>
</div>

9. Реализация алгоритма VNG. Аналоги библиотек


