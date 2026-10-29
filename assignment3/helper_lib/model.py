import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(100, 128 * 7 * 7)

        self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

def get_model(model_name):
    if model_name == "GAN":
        return Generator(), Discriminator()
    else:
        raise ValueError(f"Unsupported model: {model_name}")