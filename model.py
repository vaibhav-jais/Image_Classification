import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #############################
        # Initialize your network

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # output: 64 x 192 x 192

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # output: 64 x 192 x 192

        self.cnn_layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),     # output: 128 x 96 x 96
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))     # output: 256 x 48 x 48

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))     # output: 256 x 24 x 24

        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))      # output: 512 x 12 x 12

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 12 * 12, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(200, 80),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(80, 8)

        )

        #############################

    def forward(self, x):
        #############################
        # Implement the forward pass
        x1 = self.cnn_layer1(x)
        x2 = self.cnn_layer2(x1)
        x3 = x2 + x1
        x4 = self.cnn_layer3(x3)
        x5 = self.cnn_layer4(x4)
        x6 = x5 + x4
        x7 = self.linear_layers(x6)
        return x7
        #############################

        pass

    def save_model(self):
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################

        torch.save(self.state_dict(), 'model')