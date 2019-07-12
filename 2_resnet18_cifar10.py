"""Resnet18 Implementation on CIFAR10.

NOTE:
    - resnet 18 and 34 use BasicBlock 
    - resnet 50, 101, 152 use BottleneckBlock
    - residual layer has to be resized/downsampled before merging back
    - input size = 224
    - only 3x3 filters
    - original implementation has a dense layers
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms

DEBUG = False


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.conv_downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn_downsample = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn1(self.conv2(x))
        if self.stride == 2:
            residual = self.bn_downsample(self.conv_downsample(residual))
        x += residual
        x = self.relu(x)

        return x


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = BasicBlock(64, 64, stride=1)
        self.conv2_2 = BasicBlock(64, 64, stride=1)
        self.conv3_1 = BasicBlock(64, 128, stride=2)
        self.conv3_2 = BasicBlock(128, 128, stride=2)
        self.conv4_1 = BasicBlock(128, 256, stride=2)
        self.conv4_2 = BasicBlock(256, 256, stride=2)
        self.conv5_1 = BasicBlock(256, 512, stride=2)
        self.conv5_2 = BasicBlock(512, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def main():
    tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root='data/', train=True, transform=tfms)
    test_dataset = datasets.CIFAR10(root='data/', train=True, transform=tfms)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'Train Image Size: {images.shape}')
        print(f'Label Size: {labels.shape}')

        if DEBUG:
            plt.imshow(make_grid(images).numpy().transpose(1, 2, 0))
            plt.show()
        if batch_idx >= 0:
            break

    model = Resnet18(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters())

    # Training Loop
    for epoch in range(3):

        # Compute on Train data
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            logits, probs = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch+1} | Batch: {batch_idx} | Loss: {loss:.2f}')
            if batch_idx >= 2:
                break

        # Compute loss on Test data
        model.eval()
        for images, labels in test_loader:
            logits, probs = model(images)
            test_loss = F.cross_entropy(logits, labels)
            if batch_idx >= 2:
                print(f'Test loss: {test_loss:.2f}')
                break


if __name__ == "__main__":
    main()
