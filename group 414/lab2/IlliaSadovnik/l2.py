import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import AKAZE_create

# OpenCV matcher

def cv_matcher(first_image: np.ndarray, second_image: np.ndarray) -> List[cv2.DMatch]:
    kaze = AKAZE_create()
    first_kps, first_descs = kaze.detectAndCompute(first_image, None)
    second_kps, second_descs = kaze.detectAndCompute(second_image, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_list = matcher.match(first_descs, second_descs)
    res = cv2.drawMatches(first_image, first_kps, second_image, second_kps, matches_list, None, flags=2)
    plt.imshow(res)
    plt.title('cv_matcher')
    plt.show()
    return res


# Custom matcher

def my_matcher(first_image: np.ndarray, second_image: np.ndarray) -> List[cv2.DMatch]:
    kaze = AKAZE_create()
    first_kps, first_descs = kaze.detectAndCompute(first_image, None)
    second_kps, second_descs = kaze.detectAndCompute(second_image, None)
    matches_list = [
        cv2.DMatch(_distance=np.linalg.norm(descs1 - descs2), _imgIdx=0, _queryIdx=descs1_idx, _trainIdx=descs1_idx)
        for descs1_idx, descs1 in enumerate(first_descs) for descs1_idx, descs2 in enumerate(second_descs)
    ]
    matches_list = sorted(matches_list, key=lambda x: x.distance)[:math.floor(len(matches_list) * 0.01)]
    res = cv2.drawMatches(first_image, first_kps, second_image, second_kps, matches_list, None, flags=2)
    plt.imshow(res)
    plt.title('my_matcher')
    plt.show()
    return res


# First pair
img_1 = cv2.imread('img_1.png', cv2.IMREAD_COLOR)
img_2 = cv2.imread('img_2.png', cv2.IMREAD_COLOR)

detector = AKAZE_create()
kps1, _ = detector.detectAndCompute(img_1, None)
kps2, _ = detector.detectAndCompute(img_2, None)

plt.imshow(cv2.drawKeypoints(img_1, kps1, np.array([])))
plt.show()

plt.imshow(cv2.drawKeypoints(img_2, kps2, np.array([])))
plt.show()

cv_matcher(img_1, img_2)
my_matcher(img_1, img_2)

# Second pair
img_3 = cv2.imread('img_3.png', cv2.IMREAD_COLOR)
img_4 = cv2.imread('img_4.png', cv2.IMREAD_COLOR)

detector = AKAZE_create()
kps3, _ = detector.detectAndCompute(img_3, None)
kps4, _ = detector.detectAndCompute(img_4, None)

plt.imshow(cv2.drawKeypoints(img_3, kps3, np.array([])))
plt.show()

plt.imshow(cv2.drawKeypoints(img_4, kps4, np.array([])))
plt.show()

cv_matcher(img_3, img_4)
my_matcher(img_3, img_4)
