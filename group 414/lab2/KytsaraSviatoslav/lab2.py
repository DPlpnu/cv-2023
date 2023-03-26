import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')  # !IMPORTANT


img1 = cv.imread('img1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)

# Збільшення контрастності
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img1_contrast = clahe.apply(img1)

# Зміщення освітленості
img1_brightness = cv.convertScaleAbs(img1, alpha=1.5, beta=0)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp3, des3 = orb.detectAndCompute(img1_contrast, None)
kp4, des4 = orb.detectAndCompute(img1_brightness, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

res1 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(res1), plt.show()

res2 = cv.drawMatches(img1, kp1, img1_contrast, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(res2), plt.show()

res3 = cv.drawMatches(img1, kp1, img1_brightness, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(res3), plt.show()


def custom_match(desc1, desc2):
    matches = []
    for i, k1 in enumerate(desc1):
        for j, k2 in enumerate(desc2):
            matches.append(cv.DMatch(_queryIdx=i, _trainIdx=j, _distance=np.linalg.norm(k1 - k2, ord=2)))

    matches = sorted(matches, key=lambda x: x.distance)
    return matches


matches = custom_match(des1, des2)
res4 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(res4), plt.show()
