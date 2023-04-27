import cv2
import numpy as np

img = cv2.imread('f0a9103d9081fae3d2db758a15de460c.jpeg')
img2 = img[350:550, 100:250]

rows, cols, _ = img.shape

fast = cv2.FastFeatureDetector_create(40)
orb = cv2.ORB_create()

kp1 = fast.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
kp2 = fast.detect(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
kp1, des1 = orb.compute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kp1)
kp2, des2 = orb.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2)


# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img,None)
# kp2, des2 = sift.detectAndCompute(img2,None)


def match_descriptors(des1, des2):
    matches = []
    for i in range(len(des1)):
        dist = np.sqrt(np.sum(np.square(des1[i] - des2), axis=1))
        if np.min(dist) < 0.75:
            matches.append(cv2.DMatch(i, np.argmin(dist), np.min(dist)))
    return sorted(matches, key=lambda sample: sample.distance)


my_matches = match_descriptors(des1, des2)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda sample: sample.distance)

res1 = cv2.drawMatches(img, kp1, img2, kp2, matches[:7], None, flags=2)
res2 = cv2.drawMatches(img, kp1, img2, kp2, my_matches[:7], None, flags=2)

cv2.imshow('img', np.concatenate((res1, res2), axis=1))
cv2.waitKey(0)
cv2.destroyAllWindows()
