import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = plt.imread("tower.jpg")

plt.imshow(img1)
plt.show()

height, width = img1.shape[:2]

x_offset = 50
y_offset = 100

M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 0.8)
rotated_img = cv2.warpAffine(img1, M, (width, height))

M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
towel_match = cv2.warpAffine(rotated_img, M, (width, height))

# Відображення зображення
plt.imshow(towel_match)
plt.show()


def distanceXY(x, y):
    return sum([(x[i] ^ y[i]) for i in range(len(x))])


def matcher(image1, image2):
    orb = cv2.ORB_create()
    keypoint1, des1 = orb.detectAndCompute(image1, None)
    keypoint2, des2 = orb.detectAndCompute(image2, None)

    matches = []
    best_distance = float('inf')
    for i, k1 in enumerate(des1):
        best_match = None
        best_distance = float('inf')
        for j, k2 in enumerate(des2):
            distance = float(distanceXY(k1, k2))
            if distance < best_distance:
                best_distance = distance
                best_match = cv2.DMatch(_distance=distance, _imgIdx=0, _queryIdx=i, _trainIdx=j)

        matches.append(best_match)

    # відбираємо збіги, що задовольняють критерію
    best_matches = []
    for match in matches:
        if match.distance < 0.7 * best_distance:
            best_matches.append(match)

    result = cv2.drawMatches(image1, keypoint1, image2, keypoint2, best_matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(result)
    plt.show()


matcher(img1, towel_match)
