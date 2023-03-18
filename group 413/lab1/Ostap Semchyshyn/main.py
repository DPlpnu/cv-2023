from PIL import Image
import numpy as np

images = ['contrast1.png', 'contrast2.png', 'detail2.jpg', 'images.jpeg']
for image_filename in images:
    image = Image.open(image_filename)
    alpha = 0.1 # 0 < alpha < 1
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixels[i, j] = tuple((np.power(np.asarray(pixels[i, j]) / 255, 1 / alpha) * 255).astype(np.int32).tolist())

    image.save('processed' + image.filename)
