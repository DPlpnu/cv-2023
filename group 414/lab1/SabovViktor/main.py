import numpy as np
import matplotlib.pyplot as plt
import cv2

images = ['high_contrast.jpg', 'high_detail.jpg', 'low_contrast.jpg', 'low_detail.jpg']

def filter(img, gamma):
    result = 255 * (img/255)**(1/gamma)
    return result


def main():
    # read an image
    # 1 in the middle is the original
    gammas = [.2, .5, .8, 1, 2, 2.5, 3]

    for img_file in images:
        img = cv2.cvtColor(cv2.imread(f'./images/{img_file}'), cv2.COLOR_BGR2RGB)

        outs = list(map(lambda gamma: filter(img, gamma), gammas))

        res = np.hstack(outs) / 255
        plt.imsave(f'./images/out_{img_file}', res)


if __name__ == '__main__':
    main()

