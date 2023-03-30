import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

my_image = Image.open(
    'C:/Users/yurka/PycharmProjects/cv-2023/group 414/lab1/KurianYurii/low-contrast.jpg')


def rgb_to_gray(image):
    width, height = image.size
    gray_image = Image.new('L', (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
            gray_image.putpixel((x, y), gray)

    return gray_image


my_image = rgb_to_gray(my_image)
pix = my_image.load()
plt.imshow(my_image)
plt.show()

width, height = my_image.size
pixels = list(my_image.getdata())


def Gaus_formula(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    H = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2.0 * np.pi * sigma ** 2)
    return H


new = Gaus_formula(5)


def convolve(image, kernel):
    my_image = np.array(image)
    # Перевірка розмірів зображення та ядра
    i_rows, i_cols = my_image.shape
    k_rows, k_cols = kernel.shape
    output = np.zeros(my_image.shape)
    # Цикл по всім пікселям зображення
    for i in range(i_rows - k_rows + 1):
        for j in range(i_cols - k_cols + 1):
            # Здійснення згортки між пікселями зображення та ядром
            output[i, j] = (kernel * my_image[i:i+k_rows, j:j+k_cols]).sum()
    return output


new_image = convolve(my_image, new)

array = np.array(new_image, dtype=np.uint8)

Image.fromarray(array)

Core_x = np.asarray([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

Core_y = np.asarray([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

intensity_gradient_x = convolve(new_image, Core_x)
array = np.array(intensity_gradient_x, dtype=np.uint8)

Image.fromarray(array)
plt.imshow(array)

intensity_gradient_y =convolve(new_image, Core_y)
array = np.array(intensity_gradient_y, dtype=np.uint8)

Image.fromarray(array)
plt.imshow(array)


def magnitude(Core_x, Core_y):
    return np.hypot(Core_x, Core_y)


magnitude_threshold = magnitude(intensity_gradient_x, intensity_gradient_y)

magnitude_threshold = magnitude_threshold / magnitude_threshold.max() * 255
array = np.array(magnitude_threshold, dtype=np.uint8)

Image.fromarray(array)
plt.imshow(array)

high_bound = magnitude_threshold.max() * 0.09
low_bound = magnitude_threshold.max() * 0.09 * 0.05

high = np.where(magnitude_threshold >= high_bound, 255, 0)
low = np.where(magnitude_threshold < low_bound, 255, 0)

array = np.array(high, dtype=np.uint8)

im_b_f = Image.fromarray(array)
plt.imshow(im_b_f)
# plt.show()

array = np.array(low, dtype=np.uint8)

im_b_f = Image.fromarray(array)
plt.imshow(im_b_f)


# plt.show()


def hysteresis(image, weak=200):
    image_row, image_col = image.shape

    final_image = np.zeros((image_row, image_col))

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            if image[row, col] >= weak:
                if (image[row - 1:row + 2, col - 1:col + 2] == 255).any():
                    final_image[row, col] = 255

    return final_image


a = hysteresis(high)

array = np.array(a, dtype=np.uint8)

plt.imshow(a)
plt.show()
