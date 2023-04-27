import numpy as np
import cv2


def point_operation(img, k, l):
    img = np.asarray(img, dtype=np.float32)
    img = img * k + l
    img[img > 255] = 255
    img[img < 0] = 0
    return np.asarray(img, dtype=np.int32)


img = cv2.imread('group 412/lab1/Artemuk_Oleksandr/images/in/lis2.jpg')
filename = "lis2"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
parameters = (
    (0.5, 0),
    (1., -30),
    (1., 30),
    (1.5, 0),
    (0.5, -30),
    (0.5, 30),
)
for el in parameters:
    img = point_operation(gray, *el)
    cv2.imwrite(f"group 412/lab1/Artemuk_Oleksandr/images/out/{filename},k={el[0]},l={el[1]}.jpg", img)