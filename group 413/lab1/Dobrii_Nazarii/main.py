from PIL import Image


def check_rgb_value(value):
    return max(0, min(255, int(value)))


def apply_contrast(pixel, a, s, t):
    return tuple(check_rgb_value(a * (c - s) + t) for c in pixel)


def change_contrast(img_path, a, s, t):
    img = Image.open(img_path)
    new_pixels = [apply_contrast(pixel, a, s, t) for pixel in img.getdata()]
    new_image = Image.new(img.mode, img.size)
    new_image.putdata(new_pixels)
    return new_image


if __name__ == '__main__':
    new_image = change_contrast('photo1.jpg', a=2, s=128, t=128)
    new_image.save('result.jpg')
