from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow as tf
import numpy as np
import random
import torch

_FashionTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class AlexNetMnistFashionDataset(Dataset):
    class_names = _class_names

    def __init__(self, images, labels):
        super(AlexNetMnistFashionDataset, self).__init__()

        self.transform = _FashionTransform

        self.labels = np.asarray(labels)
        self.images = np.asarray(images).astype('float32')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.labels[idx]

        image = self.transform(self.images[idx])
            
        return image, label

    @classmethod
    def load_fashion_mnist_data(cls):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        return cls(x_train, y_train), cls(x_test, y_test)

class SiamesNetFashionMnistDataset(AlexNetMnistFashionDataset):
    
    def __init__(self, images, labels):
        super(SiamesNetFashionMnistDataset, self).__init__(images, labels)

        self.group_idxs = self.group_samples(labels)


    def group_samples(self, y):
        group_idxs = {}

        for i in range(10):
            group_idxs[i] = np.where((y==i))[0]
        
        return group_idxs

    def __getitem__(self, idx):
        image_class_1 = random.randint(0, 9)

        image_index_1 = self.group_idxs[image_class_1][random.randint(0, self.group_idxs[image_class_1].shape[0]-1)]

        if idx % 2 == 0:
            image_class_2 = image_class_1

            image_index_2 = self.group_idxs[image_class_2][self.group_idxs[image_class_2] != image_index_1][random.randint(0, self.group_idxs[image_class_1].shape[0]-2)]
        
            y = torch.tensor([1], dtype=torch.float)

        else:
            
            image_class_2 = np.arange(0, 10)[np.arange(0, 10) != image_class_1][random.randint(0, 8)]

            image_index_2 = self.group_idxs[image_class_2][random.randint(0, self.group_idxs[image_class_2].shape[0]-1)]

            y = torch.tensor([0], dtype=torch.float)
        
        image_1 = self.transform(self.images[image_index_1])

        image_2 = self.transform(self.images[image_index_2])
        
        return (image_1, image_2), y, (self.labels[image_index_1], self.labels[image_index_2])
