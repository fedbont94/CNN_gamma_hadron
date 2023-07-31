#!/usr/bin/env python3

import torch
import torch.nn as nn

# TODO Dropout? BatchNorm? Regularization?


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16 * 2, kernel_size=(3, 3, 2))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Adding dropout after the first ReLU layer
        self.conv2 = nn.Conv3d(16 * 2, 32 * 2, kernel_size=(2, 2, 1))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)  # Adding dropout after the second ReLU layer
        self.conv3 = nn.Conv3d(32 * 2, 64 * 2, kernel_size=(2, 2, 1))
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)  # Adding dropout after the third ReLU layer
        self.conv4 = nn.Conv3d(64 * 2, 64 * 2, kernel_size=(2, 2, 1))
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)  # Adding dropout after the fourth ReLU layer
        self.conv5 = nn.Conv3d(64 * 2, 128 * 2, kernel_size=(2, 2, 1))
        self.relu5 = nn.ReLU()

        # Add flattening and reduce to a single value
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 2 * 4 * 4 * 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x


# Define your FCNN architecture
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(2 + 5, 32 * 2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32 * 2)
        self.fc2 = nn.Linear(32 * 2, 64 * 2)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Adding dropout after the second ReLU layer
        self.bn2 = nn.BatchNorm1d(64 * 2)
        self.fc3 = nn.Linear(64 * 2, 128 * 2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128 * 2, 128 * 2)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)  # Adding dropout after the fourth ReLU layer
        self.bn3 = nn.BatchNorm1d(128 * 2)
        self.fc5 = nn.Linear(128 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)  # Concatenate with the other two values
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.bn3(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


# Combine the CNN and FCNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = CNN()
        self.fcnn = FCNN()

    def forward(self, qMap, tMap, fcc):
        x = self.cnn(qMap)
        y = self.cnn(tMap)
        x = torch.cat((x, y), dim=1)
        x = self.fcnn(x, fcc)
        return x
