import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img

def plot_images(images, labels) : 
    if(len(images) != len(labels)): 
        raise RuntimeError(f'Cannot assign {len(labels)} labels to {len(images)} images!') 
    fig, axes = plt.subplots(1, len(images)) 
    for idx, ax in enumerate(axes) : 
        ax.imshow(images[idx]) 
        ax.set_title(labels[idx], fontsize=12) 
        ax.axis('off') 
    plt.show()

def apply_filter(image: np.ndarray, K: int): 
    if(K < 0 or K > 255):
        raise RuntimeError('K should be between 0 and 255')
    result = image.copy() 

    # Iterate through image
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for c in range(0, image.shape[2]):
                result[i][j][c] = min([image[i][j][c] + K, 255])
    return result

images = [ 
    img.imread('high_contrast.jpg'), 
    img.imread('low_contrast.jfif'), 
    img.imread('high_detalized.jpg'), 
    img.imread('low_detalized.jpg') 
]

labels = [ 
 'Original image', 
 'Image with K=10', 
 'Image with K=50'
]

for image in images: 
    imgs = [ 
        image, 
        apply_filter(image, 10),
        apply_filter(image, 70),
    ] 
    plot_images(imgs, labels) 
