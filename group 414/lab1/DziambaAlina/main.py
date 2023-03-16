import numpy as np
import cv2


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    dist = size // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - dist)**2 + (y - dist)**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)
    return kernel


def gaussian_blur(img, kernel):
    padded_image = np.pad(img, kernel.shape[0] // 2)
    blurred_image = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            blurred_image[i, j] = np.sum(kernel * padded_image[i: i + kernel.shape[0], j: j + kernel.shape[0]])

    return blurred_image.astype(np.uint8)


def gaussian_blur_rgb(img, kernel):
    blue, green, red = cv2.split(img)
    blue = gaussian_blur(blue, kernel)
    green = gaussian_blur(green, kernel)
    red = gaussian_blur(red, kernel)
    return cv2.merge([blue, green, red])


def blur_img(img_name, kernel):
    image = cv2.imread(img_name)
    blurred_img = gaussian_blur_rgb(image, kernel)
    cv2.imwrite("output/blur_" + str(kernel.shape[0]) + "_" + img_name, blurred_img)


blur_img("high_contrast_img.jpg", gaussian_kernel(5, 1))
blur_img("high_contrast_img.jpg", gaussian_kernel(11, 2))
blur_img("low_contrast_img.jpg", gaussian_kernel(5, 1))
blur_img("low_contrast_img.jpg", gaussian_kernel(11, 2))
blur_img("high_object_detailing.jpg", gaussian_kernel(5, 1))
blur_img("high_object_detailing.jpg", gaussian_kernel(11, 2))
blur_img("low_object_detailing.jpg", gaussian_kernel(5, 1))
blur_img("low_object_detailing.jpg", gaussian_kernel(11, 2))

