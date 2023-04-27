import cv2
import numpy as np


def gamma_correction(src, gamma):
    hist_gamma = 1 / gamma

    table = [((i / 255) ** hist_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


original_img = cv2.imread('pic1.jpg')
# original_img = cv2.imread('pic2.jpg')
# original_img = cv2.imread('pic3.jpg')
# original_img = cv2.imread('pic4.jpeg')


gamma_img = gamma_correction(original_img, 0.5)

cv2.imshow('Original image', original_img)
cv2.imshow('Gamma corrected image', gamma_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
