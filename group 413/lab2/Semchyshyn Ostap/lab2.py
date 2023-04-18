import cv2
import numpy as np
from PIL import Image

img1 = cv2.imread('images_cut.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
cv2.transpose(img1, img1)

sift = cv2.xfeatures2d.DAISY_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

def match_descriptors(des1, des2):
    N1 = des1.shape[0]
    N2 = des2.shape[0]

    distances = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            distances[i, j] = np.linalg.norm(des1[i] - des2[j])

    matches = []
    for i in range(N1):
        sorted_idx = np.argsort(distances[i])
        if distances[i, sorted_idx[0]] < 0.5 * distances[i, sorted_idx[1]]:
            matches.append(cv2.DMatch(i, sorted_idx[0], distances[i, sorted_idx[0]]))

    return matches

matches = match_descriptors(des1, des2)
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
Image.fromarray(img_match.astype(np.uint8)).show()
