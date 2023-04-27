from torch.nn import functional as F
import lightning.pytorch as pl
from torch import nn
import torchmetrics
import torch

class SiamesAlexNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.featurizer = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4, 0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        self.loss = nn.BCELoss()

        self.train_accuracy = torchmetrics.Accuracy('binary')
        self.valid_accuracy = torchmetrics.Accuracy('binary')
        self.test_accuracy = torchmetrics.Accuracy('binary')

    def load_alexnet_featurizer(self, alexnet_state_dict):
        siames_state_dict = self.state_dict()

        alexnet_features_state_dict = {k: v for k, v in alexnet_state_dict.items() if 'featurizer' in k}

        siames_state_dict.update(alexnet_features_state_dict)

        self.load_state_dict(siames_state_dict)

        for param in self.featurizer.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_features(self, x):
        x = self.featurizer(x)

        return x.view(x.shape[0], -1)
    
    def forward(self, x):
        x_1, x_2 = x

        features_1 = self.get_features(x_1)
        features_2 = self.get_features(x_2)

        concat_features = torch.cat((features_1, features_2), 1)

        return self.classifier(concat_features)
    

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.train_accuracy(y_pred, y)

        self.log('Train loss', loss, on_epoch=True, on_step=False)
        self.log('Train acc', acc, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.valid_accuracy(y_pred, y)

        self.log('Validation loss', loss, on_epoch=True, on_step=False)
        self.log('Validation acc', acc, on_epoch=True, on_step=False)

        return loss
    
    def testing_step(self, batch, batch_idx):
        x, y, _ = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.test_accuracy(y_pred, y)

        self.log('Test loss', loss, on_epoch=True, on_step=False)
        self.log('Test acc', acc, on_epoch=True, on_step=False)

        return loss
