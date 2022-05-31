## Лабораторная работа 2. Переход между цветовыми пространствами. Линейный и нелинейный переход. Мера цветовой разницы. Функции преобразования яркости. Гамма, логарифмическое, экспоненциаяльное кодирование.

1. Скачать любое цифровое изображение. Желательно многоцветное

<div align="center">
<div>Исходное изображение:</div>
 <img src="img1.jpg" alt="" width=66%>
</div>
2. Отобразить изображение по каналам RGB (каждый канал представить как градации серого).

```
img = cv2.imread("../img.jpg")
b, g, r = cv2.split(img)

cv2.imwrite("r.jpg", r)
cv2.imwrite("b.jpg", b)
cv2.imwrite("g.jpg", g)
```

<div align="center">
 <div>R:</div>
 <img src="part%201/r.jpg" width=66% alt="">
  <div>G:</div>
 <img src="part%201/g.jpg" width=66% alt="">
  <div>B:</div>
 <img src="part%201/b.jpg" width=66% alt="">
</div>

3. Лианеризовать изображение обратным гамма преобразованием.

```
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


img0 = gamma_trans(img, 1 / 2.2)
cv2.imwrite("out.jpg", img0)
```

<div align="center">
 <div>Результат:</div>
 <img src="part%201/out.jpg" width=66% alt="">
</div>

4. Отобразить по каналам RGB.

```
def get_channels(image):
    red, green, blue = image.copy(), image.copy(), image.copy()
    red[:, :, (1, 2)] = 0
    green[:, :, (0, 2)] = 0
    blue[:, :, (0, 1)] = 0
    return red, green, blue


r, g, b = get_channels(img0)
cv2.imwrite("outR.jpg", r)
cv2.imwrite("outG.jpg", g)
cv2.imwrite("outB.jpg", b)
```

<div style="display: inline-flex">
    <div align="center"> 
        <div>R:</div>
         <img src="part%201/outR.jpg" alt="">
    </div>
    <div align="center"> 
        <div>G:</div>
        <img src="part%201/outG.jpg" alt="">
    </div>
    <div align="center"> 
        <div>B:</div>
        <img src="part%201/outB.jpg" alt="">
    </div>
</div>

5. Отобразить поканально разницу между исходным изображением и линеаризованным.

<div align="center">
<div>Разница:</div>
 <img src="part%201/dif.jpg" alt="" width=66%>
</div>

6. Написать функцию перевода цветов из линейного RGB в XYZ с использованием матрицы. Найти подходящую библиотечную
   функцию. Сравнить результаты через построение разностного изоборажения.

```
# перевод с использованием матрицы
matrix_RGB_to_XYZ = np.array(
    [[0.49000, 0.31000, 0.20000],
     [0.17697, 0.81240, 0.01063],
     [0.00000, 0.01000, 0.99000]]
)


def rgb_to_xyz(image):
    r0, g0, b0 = cv2.split(image)
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))

XYZ = rgb_to_xyz(img)
```

<div align="center">
 <img src="part%202/img_1.png" alt="" width=66%>
</div>

```
# перевод с использованием библиотечной функции skimage
XYZ_skimage = skimage.color.rgb2xyz(RGB.astype(np.uint8))
```

<div align="center">
 <img src="part%202/img.png" alt="" width=66%>
</div>

<div align="center">
<div>Разностное изоборажение:</div>
 <img src="part%202/dif.jpg" alt="" width=66%>
</div>

7. Написать функцию перевода цветов из XYZ в RGB (построить обратную матрицу XYZ в RGB). Преобразовать изображение XYZ в
   линейный RGB. Применить гамма преобразование. Сравнить результаты через построение разностного изоборажения.

```
# перевод с использованием обратной матрицы
matrix_XYZ_to_RGB = np.array(
    [[2.36461385, -0.89654057, -0.46807328],
     [-0.51516621, 1.4264081, 0.0887581],
     [0.0052037, -0.01440816, 1.00920446]]
)


def xyz_to_rgb(image):
    r0, g0, b0 = cv2.split(image)
    r = r0 / 255.0
    g = g0 / 255.0
    b = b0 / 255.0
    xyz_x = matrix_XYZ_to_RGB[0][0] * r + matrix_XYZ_to_RGB[0][1] * g + matrix_XYZ_to_RGB[0][2] * b
    xyz_y = matrix_XYZ_to_RGB[1][0] * r + matrix_XYZ_to_RGB[1][1] * g + matrix_XYZ_to_RGB[1][2] * b
    xyz_z = matrix_XYZ_to_RGB[2][0] * r + matrix_XYZ_to_RGB[2][1] * g + matrix_XYZ_to_RGB[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))


RGB = xyz_to_rgb(img)
```

<div align="center">
 <img src="part%202/img_3.png" alt="" width=66%>
</div>

```
# перевод с использованием библиотечной функции skimage
RGB_skimage = skimage.color.xyz2rgb(img.astype(np.uint8))
```

<div align="center">
 <img src="part%202/img_7.png" alt="" width=66%>
</div>

<div align="center">
<div>Разностное изоборажение между переведенным в RGB через обратную матрицу преобразования XYZ в RGB и через библиотечную функцию:</div>
 <img src="part%202/difRGBLib.jpg" alt="" width=66%>
</div>

8. Построить проекцию цветов исходного изображения на цветовой локус (плоскость xy).

```
X = np.array(rgb_to_xyz(img)[:, 1])
Y = np.array(rgb_to_xyz(img)[:, 2])
Z = np.array(rgb_to_xyz(img)[:, 3])
node = 200
array = np.array([[0.4898, 0.3101, 0.2001],
                  [0.1769, 0.8124, 0.0107],
                  [0.0000, 0.0100, 0.9903]])
InvArray = np.linalg.inv(array)
x = X / (X + Y + Z)
y = Y / (X + Y + Z)
z = Z / (X + Y + Z)
xs, ys, zs = 0, 0, 0
tmpx = np.linspace(x[0], x[-1], node)
tmpy = np.linspace(y[0], y[-1], node)
for i in range(len(x)):
    xs += x[i]
    ys += y[i]
    zs += z[i]

XWhite = xs / (xs + ys + zs)
YWhite = ys / (xs + ys + zs)
Allx, Ally, AllX, AllZ = [], [], [], []
x = np.append(x, tmpx)
y = np.append(y, tmpy)
for (v, v2) in zip(x, y):
    tmpx = np.linspace(XWhite, v, node)
    Allx = np.append(Allx, tmpx)
    tmpy = np.linspace(YWhite, v2, node)
    Ally = np.append(Ally, tmpy)

AllY = Ally
AllX = Allx
AllZ = 1 - Allx - Ally
XYZ = [[x, y, z] for (x, y, z) in zip(AllX, AllY, AllZ)]
RGB = [np.dot(InvArray, v) for v in XYZ]
for v in RGB:
    if (v[0] < 0):
        v[0] = 0
    if (v[0] > 1):
        v[0] = 1
    if (v[1] < 0):
        v[1] = 0
    if (v[1] > 1):
        v[1] = 1
    if (v[2] < 0):
        v[2] = 0
    if (v[2] > 1):
        v[2] = 1

plt.figure()
plt.scatter(Allx, Ally, c=RGB)
plt.show()
```

<div align="center">
 <div>Проекция цветов на цветовой локус (плоскость xy):</div>

![img.png](img.png)

</div>

9. Написать функцию перевода цветов из линейного RGB в HSV и обратно. Найти подходящую библиотечную функцию. Сравнить
   результаты через построение разностного изоборажения.

```
def rgb2hsv(rgb):
    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[:, :, 1] - rgb[:, :, 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[:, :, 2] - rgb[:, :, 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[:, :, 0] - rgb[:, :, 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[:, :, 2] = maxv

    return hsv
```

<div align="center">
 <img src="part%204/img_1.png" alt="" width=66%>
</div>

```
# перевод с использованием библиотечной функции skimage
HSV_skimage = skimage.color.rgb2hsv(RGB.astype(np.uint8))
```

<div align="center">
 <img src="part%204/img.png" alt="" width=66%>
</div>

<div align="center">
<div>Разностное изоборажение между ручным преобразованием из RGB в HSV и преобразованием через библиотечную функцию:</div>
 <img src="part%204/difHSV.png" alt="" width=66%>
</div>

### Функция перевода цветов из HSV в RGB

```
def hsv2rgb(hsv):
    hi = np.floor(hsv[:, :, 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[:, :, 2].astype('float')
    f = (hsv[:, :, 0] / 60.0) - np.floor(hsv[:, :, 0] / 60.0)
    vmin = v * (1.0 - hsv[:, :, 1])
    vdec = v * (1.0 - (f * hsv[:, :, 1]))
    vinc = v * (1.0 - ((1.0 - f) * hsv[:, :, 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, vinc, vmin))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((vdec, v, vmin))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((vmin, v, vinc))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((vmin, vdec, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((vinc, vmin, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, vmin, vdec))[hi == 5, :]
    return rgb
```

<div align="center">
 <img src="part%204/RGBNew.png" alt="" width=66%>
</div>

```
# перевод с использованием библиотечной функции skimage
RGBNew_skimage = skimage.color.hsv2rgb(HSV_skimage)
```

<div align="center">
 <img src="part%204/img_2.png" alt="" width=66%>
</div>

11. Используя библиотечные функции цветовой разности сравнить результаты, полученные в пунктах 6, 7, 9 (для каждой
    функции).


