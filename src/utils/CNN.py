import torch
from torchvision.models import resnet18
from utils.modules import v2iDownLayer

class CNN(torch.nn.Module):
    def __init__(self, inputDim, outputDim, num_downLayer=2):
        super().__init__()
        self.v2i = v2iDownLayer(inputDim, num_downLayer)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.ReLU(),
        )
        self.resnet = resnet18()
        self.resnet.fc = torch.nn.Linear(512, outputDim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.v2i(x)
        out = self.conv(out)
        out = self.resnet(out)
        out = self.softmax(out)
        return out

class CNN_(torch.nn.Module):
    def __init__(self, inputDim, outputDim, num_downLayer=3):
        super().__init__()
        self.v2i = v2iDownLayer(inputDim, num_downLayer)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Linear(64, outputDim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.v2i(x)
        out = self.conv(out)
        out = self.pool(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out
