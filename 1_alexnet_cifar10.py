"""Alexnet Implementation on CIFAR10.

NOTE:
    - inout size = 224
    - 3 filters sizes: 11, 5, 3 with different padding and strides
    - 3 dense layers

TODO:
  - [x] Use CIFAR10 from Torchvision
  - [x] Use dataloaders from Torch
  - [x] Use Transforms for Resize from Torch
  - [x] Visualize batch before training
  - [x] Model Summary
  - [x] Manual training loop
  - [x] Implement Accuracy metric
  - [x] Evaluate on Test data
  - [x] Run On GPU/CPU
  - [x] Save model to disk
  - [ ] Inference script
  - [ ] Running average of loss
"""
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchsummary import summary

DEBUG_BATCH = False
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


def imshow_batch(images_batch, labels_batch_names):
    # make_grid is a Pytorch util function that take a batch and makes it one long tensor
    imshow(make_grid(images_batch), labels_batch_names)


def imshow(img, labels_batch_names):
    plt.text(0, 0, ' '.join(labels_batch_names))

    img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()


def compute_accuracy(model, dataloader):
    correct_samples, total_samples = 0, 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        logits, probs = model(images)
        predicted_labels = torch.max(probs, dim=1)[1]  # [1] gives argmax
        assert predicted_labels.size() == labels.size()
        correct_samples += (predicted_labels == labels).sum()
        total_samples += len(predicted_labels)
        if batch_idx == 5:
            break
    accuracy = float(correct_samples) * 100 / total_samples
    return accuracy


class Alexenet(nn.Module):
    def __init__(self, num_classes):
        super(Alexenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max_pool2d(self.relu(self.conv5(x)))

        x = x.view(-1, (256 * 6 * 6))
        x = self.relu(self.fc6(self.dropout(x)))
        x = self.relu(self.fc7(self.dropout(x)))

        logits = self.fc8(x)
        probs = F.softmax(logits, dim=1)

        return logits, probs


def main():
    # resize & numpy -> torch tensor transforms
    transform_list = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # create datasets
    train_dataset = CIFAR10(root='data/', train=True, transform=transform_list, download=True)
    test_dataset = CIFAR10(root='data/', train=False, transform=transform_list, download=True)
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)  # noqa

    # visualize single batch of data
    images_batch, labels_batch = next(iter(train_loader))
    print(f'Image batch dimensions: {images_batch.size()}')
    print(f'Image label dimensions: {labels_batch.size()}')

    if DEBUG_BATCH:
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        labels_batch_names = [idx_to_class[x] for x in labels_batch.numpy()]  # noqa
        imshow_batch(images_batch, labels_batch_names)

    # Instantiate Model
    model = Alexenet(num_classes=10)
    model.to(DEVICE)
    print(summary(model, input_size=(3, 224, 224)))

    optimizer = torch.optim.Adam(model.parameters())

    # Traning Loop
    for epoch in range(3):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits, probas = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(f'Epoch: {epoch+1} | Batch: {batch_idx} | Loss: {loss:.2f}')
            if batch_idx == 5:
                break
        # Evaluate on Test data
        model.eval()
        with torch.no_grad():
            train_accuracy = compute_accuracy(model, train_loader)
            test_accuracy = compute_accuracy(model, test_loader)
            print(f'Train Accuracy: {train_accuracy:.2f} | ' f'Test Accuracy: {test_accuracy:.2f}')

    # Save the model to disk
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'out/alexnet.pth'
    )

    print('Done...')


if __name__ == "__main__":
    main()
