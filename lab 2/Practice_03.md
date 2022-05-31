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

```

<div align="center">
<div>Изображение:</div>
 <img src="" alt="" width=66%>
</div>

5. Отобразить поканально разницу между исходным изображением и линеаризованным.

<div align="center">
<div>Разница:</div>
 <img src="part%201/dif.jpg" alt="" width=66%>
</div>

6. Написать функцию перевода цветов из линейного RGB в XYZ с использованием матрицы. Найти подходящую библиотечную функцию. Сравнить результаты через построение разностного изоборажения.
```
def rgb_to_xyz(image):
    r0, g0, b0 = cv2.split(image)
    r = 1 / 2.2 * r0 / 255.0
    g = 1 / 2.2 * g0 / 255.0
    b = 1 / 2.2 * b0 / 255.0
    xyz_x = matrix_RGB_to_XYZ[0][0] * r + matrix_RGB_to_XYZ[0][1] * g + matrix_RGB_to_XYZ[0][2] * b
    xyz_y = matrix_RGB_to_XYZ[1][0] * r + matrix_RGB_to_XYZ[1][1] * g + matrix_RGB_to_XYZ[1][2] * b
    xyz_z = matrix_RGB_to_XYZ[2][0] * r + matrix_RGB_to_XYZ[2][1] * g + matrix_RGB_to_XYZ[2][2] * b
    return dstack((xyz_x, xyz_y, xyz_z))

XYZ = rgb_to_xyz(img)
```
<div align="center">
<div>Разностное изоборажение:</div>
 <img src="part%202/dif.jpg" alt="" width=66%>
</div>

7. Написать функцию перевода цветов из XYZ в RGB (построить обратную матрицу XYZ в RGB). Преобразовать изображение XYZ в линейный RGB. Применить гамма преобразование. Сравнить результаты через построение разностного изоборажения.

<div align="center">
 <img src="" alt="" width=66%>
</div>

8. Построить проекцию цветов исходного изображения на цветовой локус (плоскость xy).

```

```
<div>
 <div>:</div>
 <img src="" alt="" width=66%>
</div>

9. Написать функцию перевода цветов из линейного RGB в HSV и обратно. Найти подходящую библиотечную функцию. Сравнить результаты через построение разностного изоборажения.
10. Используя библиотечные функции цветовой разности сравнить результаты, полученные в пунктах 6, 7, 9 (для каждой функции).


