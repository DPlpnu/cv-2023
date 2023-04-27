import cv2
import numpy as np

images = [
    # 'f111d3e7-df6e-4a31-a4e9-3bce6c8a89fe.jpeg',
    'LosColoresdePatricia.jpeg'
]


def use_kernel(size: int, img):
    kernel = np.ones((size, size), np.float32) / (size ** 2)
    slide = size // 2

    try:
        res_img = img
        for row in range(slide, len(img) - slide):
            for col in range(slide, len(img[row]) - slide):

                img_part = [_[col - slide: col + size - slide] for _ in img[row - slide: row + size-slide]]
                res_img[row][col] = [sum(map(sum,[[img_part[r][c][color] * kernel[r][c] for c in range(len(img_part[r]))] for r in range(len(img_part))])) for color in range(3)]

        print(f'Processing for img with kernel({size}*{size}) - done!')
        return res_img
    except Exception:
        print('Sth went wrong:', Exception)
        return img


def filter_2d(img_path: str):
    img = cv2.imread(img_path)
    imgs = [img] + [use_kernel(i, cv2.imread(img_path)) for i in [5, 7, 9]]
    res = np.concatenate(imgs, axis=1)
    cv2.imshow(img_path, res)


if __name__ == '__main__':
    for img in images:
        filter_2d(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
