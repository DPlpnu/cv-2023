from PIL import Image


def change_contrast(img_path: str, a: float, s: int, t: int) -> Image:
    def _check_rgb_condition(value: int) -> int:
        return 0 if value < 0 else 255 if value > 255 else int(value)

    def _change_pixel(p: tuple) -> tuple:
        return tuple(_check_rgb_condition(a * (c - s) + t) for c in p)

    img = Image.open(img_path)

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            img.putpixel((x, y), _change_pixel(img.getpixel((x, y))))

    return img


if __name__ == '__main__':
    new_image = change_contrast('test_image_output.jpg', a=2, s=128, t=128)
    new_image.save('test_image_output1.jpg')
