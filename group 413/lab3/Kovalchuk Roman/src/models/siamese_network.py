from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights


class SiameseNetwork(nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer.
    The output of the linear layer is passed through a sigmoid function.
    This implementation uses the GoogLeNet model with pretrained weights from ImageNet.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Get GoogLeNet model with pretrained weights
        self.googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

        # Freeze learning in CNN layer
        for param in self.googlenet.parameters():
            param.requires_grad = False

        # Replace the last classification layer with a linear layer
        self.fc_in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(self.fc_in_features, 32)

        # Add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(32 * 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def __forward_cnn(self, x):
        output = self.googlenet(x)
        output = output.view(output.size()[0], -1)

        return output

    def forward(self, x, y):
        # Get two images' features
        output1 = self.__forward_cnn(x=x)
        output2 = self.__forward_cnn(x=y)

        # Concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # Pass the concatenation to the linear layers
        output = self.fc(output)

        return output

    def to(self, *args, **kwargs):
        self.googlenet = self.googlenet.to(*args, **kwargs)
        result = super().to(*args, **kwargs)

        return result
