import matplotlib.pyplot as plt
import numpy as np
import os

# Завдання:
# Виконати гістограмне збільшення гамми (див. лекція №1). Провести порівняльний  аналіз 


def absoluteFilePath(file_name):
    # Повертає абсолютний шлях до файлу
    return os.path.dirname(__file__) + '/' + file_name

def gamma_correction(img, gamma):
    # Виконуємо гістограмне збільшення гамми
    return (255 * np.power(img / 255, 1 / gamma)).astype(np.uint8)

def show_comparison(imgs, imgs_gamma, names):
    # Відображаємо оригінальне зображення та зображення з новим гамма-коефіцієнтом
    length = len(imgs) # Кількість зображень

    _, axs = plt.subplots(length, 2, squeeze=False, figsize=(18, 10)) # Створюємо полотно для відображення зображень

    for i in range(length):
        axs[i][0].imshow(imgs[i]) # Відображаємо оригінальне зображення
        axs[i][0].set_title('Original ' + names[i]) # Встановлюємо заголовок
        axs[i][0].axis('off') # Вимикаємо відображення осей

        axs[i][1].imshow(imgs_gamma[i]) # Відображаємо зображення з новим гамма-коефіцієнтом
        axs[i][1].set_title('Gamma corrected ' + names[i]) # Встановлюємо заголовок
        axs[i][1].axis('off') # Вимикаємо відображення осей
    plt.show() # Відображаємо полотно

# Гамма-коефіцієнт
gamma = 1.8

# Зображення для перевірки
different_quality_img_names = ['low_quality.jpg', 'high_quality.jpg']

different_contrast_img_names = ['low_contrast.jpg', 'high_contrast.jpg']

# Зчитуємо зображення
different_quality_imgs = [plt.imread(absoluteFilePath(name)) for name in different_quality_img_names]

different_contrast_imgs = [plt.imread(absoluteFilePath(name)) for name in different_contrast_img_names]

# Застосовуємо гамма-корекцію
different_quality_imgs_gamma = [gamma_correction(img, gamma) for img in different_quality_imgs]

different_contrast_imgs_gamma = [gamma_correction(img, gamma) for img in different_contrast_imgs]

# Відображаємо зображення
show_comparison(different_quality_imgs, different_quality_imgs_gamma, different_quality_img_names)

show_comparison(different_contrast_imgs, different_contrast_imgs_gamma, different_contrast_img_names)
