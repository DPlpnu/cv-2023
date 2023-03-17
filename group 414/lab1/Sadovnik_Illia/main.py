import copy

import cv2
import numpy as np


def filter2d(image_array: np.array, kernel_array: np.array):

    result = np.zeros_like(image_array)
    copy_image = copy.deepcopy(image_array)

    for i in range(kernel_array.shape[0]+1):
        copy_image = np.pad(copy_image, [(0, 1), (0, 1)], mode='constant')

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            result[x][y] = (np.array(copy_image[x:(x+kernel_array.shape[0]), y: (y+kernel_array.shape[1])])
                            * np.array(kernel_array)).sum()

    return result


# read image

image = cv2.imread('img.png',0)

cv2.imshow('image', image)
cv2.waitKey(0)


# Roberts

kernel_Roberts_y = np.array([[0, -1],
                             [1, 0]])
kernel_Roberts_x = np.array([[-1, 0],
                             [0, 1]])


# Sobel


kernel_Sobel_x = np.array([[-10, 0, 1],
                           [-2, 0, 2],
                           [-10, 0, 1]])
kernel_Sobel_y = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])


# filter2d


output = filter2d(image, kernel_Sobel_y)


# output


cv2.imshow('image', output)
cv2.waitKey(0)


# test

# output = cv2.filter2D(image, -1, kernel_Sobel_y)
# cv2.imshow('image', output)
# cv2.waitKey(0)