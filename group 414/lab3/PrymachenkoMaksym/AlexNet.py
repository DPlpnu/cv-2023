import torch.nn.functional as F
import lightning.pytorch as pl
from torch import nn
import torchmetrics
import torch



class AlexNet(pl.LightningModule):

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
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)
        self.valid_accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_features(self, x):
        x = self.featurizer(x)

        return x.view(x.shape[0], -1)

    def forward(self, x):

        return self.classifier(self.get_features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.train_accuracy(y_pred, y)

        self.log('Train loss', loss, on_epoch=True, on_step=False)
        self.log('Train acc', acc, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.valid_accuracy(y_pred, y)

        self.log('Validation loss', loss, on_epoch=True, on_step=False)
        self.log('Validation acc', acc, on_epoch=True, on_step=False)

        return loss
    
    def testing_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        acc = self.test_accuracy(y_pred, y)

        self.log('Test loss', loss, on_epoch=True, on_step=False)
        self.log('Test acc', acc, on_epoch=True, on_step=False)

        return loss
