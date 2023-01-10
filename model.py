import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #############################
        # Initialize your network
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride=1, padding=1),               # 16*128*128
            # implementing Batch normalization at the end of each layer
            nn.BatchNorm2d(16),                                                   # 16*128*128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),                # 64*128*128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))                                                # 64*128*128
            #nn.MaxPool2d(kernel_size=2, stride=2))                               # output: 64 x 64 x 64

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))                                                # 64*128*128


        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),                                                # 128*128*128
            nn.MaxPool2d(kernel_size=2, stride=2))                                # output: 128 x 64 x 64

        self.cnn_layer4 =  nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))                                                # 256*64*64
            #nn.MaxPool2d(kernel_size=2, stride=2))                               # output: 256 x 16 x 16

        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))                                                # 256*64*64
            
        self.cnn_layer6 =  nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))                                # output: 512 x 32 x 32

        self.cnn_layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))                                                # 512*32*32

        self.cnn_layer8 =  nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))                                # output: 512 x 16 x 16

        self.cnn_layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))                                # output: 512 x 8 x 8

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*8*8, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(200, 80),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(80, 8))

        #############################
        
    def forward(self, x):

        #############################
        # Implement the forward pass
        x1 = self.cnn_layer1(x)
        x2 = self.cnn_layer2(x1)
        x3 = x2 + x1
        x4 = self.cnn_layer3(x3)
        x5 = self.cnn_layer4(x4)
        x6 = self.cnn_layer5(x5)
        x7 = x6 + x5
        x8 = self.cnn_layer6(x7)
        x9 = self.cnn_layer7(x8)
        x10 = x9 + x8
        x11 = self.cnn_layer8(x10)
        x12 = self.cnn_layer9(x11)
        
        #x5 = x5.view(x.size(0), -1)
        x13 = self.linear_layers(x12)
        return x13
        #############################

        pass

    def save_model(self):

        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################

        torch.save(self.state_dict(), 'model_CNN+Adam+Droupout_again')                        # saves and loads only the model parameters
