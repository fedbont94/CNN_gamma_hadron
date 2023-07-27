#!/usr/bin/env python3
import torch
import torch.nn as nn


# Define your CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16 * 2, kernel_size=(3, 3, 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(16 * 2, 32 * 2, kernel_size=(2, 2, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(32 * 2, 64 * 2, kernel_size=(2, 2, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(64 * 2, 64 * 2, kernel_size=(2, 2, 1))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv3d(64 * 2, 128 * 2, kernel_size=(2, 2, 1))
        self.relu5 = nn.ReLU()

        # Add flattening and reduce to a single value
        self.faltten = nn.Flatten()
        self.fc = nn.Linear(128 * 2 * 4 * 4 * 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.faltten(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x


# Define your FCNN architecture
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(1 + 4, 32 * 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32 * 2, 64 * 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64 * 2, 128 * 2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128 * 2, 128 * 2)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)  # Concatenate with the other two values
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


# Combine the CNN and FCNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = CNN()
        self.fcnn = FCNN()

    def forward(self, x, y):
        x = self.cnn(x)
        x = self.fcnn(x, y)
        return x
