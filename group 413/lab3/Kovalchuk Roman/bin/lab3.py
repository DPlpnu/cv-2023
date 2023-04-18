import torch
from tqdm import tqdm

from src.models.siamese_network import SiameseNetwork

model = SiameseNetwork()
model = model.to(device='mps', dtype=torch.float16)
model
#%%
next(model.parameters()).device, next(model.parameters()).dtype
#%%
import torchinfo

input_size = (1, 3, 64, 64)

summary = torchinfo.summary(
    model=model.to(device='cpu', dtype=torch.float32),
    input_size=[input_size, input_size],
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)
summary
#%%
first_parameter = next(model.parameters())
input_shape = first_parameter.size()
input_shape
#%%
model = model.eval()
#%%
import torch
from torch import optim
from src.losses.contrastive_loss import ContrastiveLoss
from torch.utils.data import DataLoader
from src.datasets.siamese_dataset import SiameseDataset
from torchvision import datasets
from torchvision.transforms import transforms

training_percent = 0.8
manual_seed = 42

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root='../data', train=True, transform=transform, download=False)
test_dataset = datasets.FashionMNIST(root='../data', train=False, transform=transform, download=False)

train_siamese_dataset = SiameseDataset(train_dataset)
test_siamese_dataset = SiameseDataset(test_dataset)

training_samples = int(len(train_siamese_dataset) * training_percent)
validation_samples = len(train_siamese_dataset) - training_samples

train_set, val_set = torch.utils.data.random_split(
    train_siamese_dataset,
    [training_samples, validation_samples],
    generator=torch.Generator().manual_seed(manual_seed),
)

# Create Siamese DataLoaders
train_siamese_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_siamese_loader = DataLoader(val_set, batch_size=8, shuffle=True)
test_siamese_loader = DataLoader(test_siamese_dataset, batch_size=10, shuffle=False)
#%%
import matplotlib.pyplot as plt
import torchvision.utils as vutils

for x, y, t in test_siamese_loader:
    for item in range(x.size(0)):
        input1 = x[item]
        input2 = y[item]
        target = t[item]

        # Plot the input1 and input2 images
        plt.figure(figsize=(4, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(vutils.make_grid(input1, nrow=5, normalize=True).permute(1, 2, 0))
        plt.axis('off')
        plt.title('Input 1 (Target: {})'.format(target.item()))
        plt.subplot(1, 2, 2)
        plt.imshow(vutils.make_grid(input2, nrow=5, normalize=True).permute(1, 2, 0))
        plt.axis('off')
        plt.title('Input 2 (Target: {})'.format(target.item()))
        plt.show()

    break
#%%
from torch import nn
from torch.nn import BCELoss
from src.constants.training_constants import TrainingConstants
from datetime import datetime
from src.training_utils.early_stopper import EarlyStopper

tolerance = 10
min_delta = 1e-2
n_epochs = 1

model.train()
model = model.to(device='mps', dtype=torch.float32)
early_stopper = EarlyStopper(
    tolerance=tolerance,
    min_delta=min_delta,
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()

for epoch in tqdm(range(n_epochs), desc='Epoch'):
    loss_train = 0.0
    loss_val = 0.0
    total = 0

    for x, y, t in tqdm(train_siamese_loader, desc='Train', leave=False):
        x = x.repeat(1, 3, 1, 1).to(device='mps', dtype=torch.float32)
        y = y.repeat(1, 3, 1, 1).to(device='mps', dtype=torch.float32)
        t = t.half().to(device='mps', dtype=torch.float32)

        optimizer.zero_grad()
        output = model(x=x, y=y)
        loss = loss_fn(output, t)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    with torch.no_grad():
        for x, y, t in tqdm(val_siamese_loader, desc='Val', leave=False):
            x = x.repeat(1, 3, 1, 1).to(device='mps', dtype=torch.float32)
            y = y.repeat(1, 3, 1, 1).to(device='mps', dtype=torch.float32)
            t = t.half().to(device='mps', dtype=torch.float32)

            output = model(x=x, y=y)
            loss = loss_fn(output, t)

            loss_val += loss.item()

    train_loss = loss_train / len(train_siamese_loader)
    val_loss = loss_val / len(val_siamese_loader)

    message = TrainingConstants.EPOCH_MESSAGE.format(
        time=datetime.now(),
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
    )

    print(message)