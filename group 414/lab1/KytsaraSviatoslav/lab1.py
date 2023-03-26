# import cv2
# import numpy as np
#
# image = cv2.imread("img.jpeg")
# cv2.imshow("Original", image)
# gamma = [0.2, 0.5, 0.7]
# for i in gamma:
#     invGamma = 1.0 / i
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     res_img = cv2.LUT(image, table)  # змінює значення пікселів
#     cv2.imshow(f"Gamma {i}", res_img)
#
# cv2.waitKey(0)




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')  # !IMPORTANT


image = plt.imread('img.jpeg')
plt.imshow(image), plt.show()
gamma = [0.2, 0.5, 0.7]
for i in gamma:
    img = image / 255.0
    reduced_img = np.power(img, 1 / i)
    reduced_img = reduced_img * 255.0
    reduced_img = np.uint8(reduced_img)
    plt.imshow(reduced_img), plt.show()
