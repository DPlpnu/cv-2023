import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img


def change_brightness(image: np.ndarray, K: int):
    new_img = image.copy()

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for c in range(0, image.shape[2]):
                new_val = image[i][j][c] + K
                if new_val > 255:
                    new_val = 255
                elif new_val < 0:
                    new_val = 0
                new_img[i][j][c] = new_val
    return new_img


def show_images(images, labels):
    fig, axes = plt.subplots(1, len(images))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx])
        ax.set_title(labels[idx], fontsize=20)
        ax.axis('off')
    plt.show()


images = [
    img.imread('high_contrast.jpg'),
    img.imread('low_contrast.jpg'),
    img.imread('high_resolution.jfif'),
    img.imread('low_resolution.jfif')
]

labels = [
    'Original',
    'K=-20',
    'K=-100'
]

for image in images:
    imgs = [
        image,
        change_brightness(image, -20),
        change_brightness(image, -100),
    ]
    show_images(imgs, labels)
