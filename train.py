#!/usr/bin/env python
"""
Script to generate the weights of the model that predicts digits. The script flow is:

download mnist data -> train cnn model -> save the weights -> evaluate the model

Note: the script works with or without GPU support

"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from architecture import create_cnn_model

WEIGHTS_PATH = "./weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int = 32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = load_data(batch_size=32)

    net = create_cnn_model().to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training Loop
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}")
                running_loss = 0.0

    print("Finished training!")

    Path(WEIGHTS_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(WEIGHTS_PATH, "mnist_net.pth"))

    # Evaluation

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
