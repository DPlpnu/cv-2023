import numpy as np
import cv2 as cv
import os

FILE_NAME = 'contrast'


def absoluteFilePath(file_name):
    return os.path.dirname(__file__) + '/' + file_name


img1 = cv.imread(absoluteFilePath(FILE_NAME + '.jpg'),
                 cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread(absoluteFilePath(FILE_NAME + '_in_scene.jpg'),
                 cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate BRIEF detector
star = cv.xfeatures2d.StarDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints and descriptors with BRIEF

kp1 = star.detect(img1, None)
kp2 = star.detect(img2, None)
# compute the descriptors with BRIEF
kp1, des1 = brief.compute(img1, kp1)
kp2, des2 = brief.compute(img2, kp2)
print(des1.shape)
print(des2.shape)


def normHamming(des1, des2):
    if len(des1) != len(des2):
        raise ValueError('Input descriptors have different lengths')

    distance = 0
    for d1, d2 in zip(des1, des2):
        distance += bin(d1 ^ d2).count('1')

    return distance/len(des1)


def threshold_matches(matches, threshold):
    # good_matches = []
    # for m in matches:
    #     if m.distance < threshold:
    #         good_matches.append(m)
    # return good_matches

    return sorted(matches, key=lambda x: x.distance)[:threshold]


# Match descriptors using a custom matching function
def custom_match(des1, des2, threshold):
    # Create a BFMatcher object
    matches = []
    for i, d1 in enumerate(des1, start=0):
        for j, d2 in enumerate(des2, start=0):
            matches.append(cv.DMatch(_trainIdx=j, _queryIdx=i,
                           _distance=normHamming(d1, d2), _imgIdx=0))

    return threshold_matches(matches, threshold)


def cv_match(des1, des2, threshold):
    # Create a BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    return threshold_matches(matches, threshold)


img3 = cv.drawMatches(img1, kp1, img2, kp2, cv_match(
    des1, des2, 25), None, flags=2)
img4 = cv.drawMatches(img1, kp1, img2, kp2, custom_match(
    des1, des2, 25), None, flags=2)

cv.imwrite(absoluteFilePath(FILE_NAME + '_cv_match.jpg'), img3)
cv.imwrite(absoluteFilePath(FILE_NAME + '_custom_match.jpg'), img4)
