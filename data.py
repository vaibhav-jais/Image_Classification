import torch
import torch.nn.functional as F
import torchvision # this package consists of popular datasets, model architectures and common image transformations for CV
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision import models, transforms

import numpy as np
import pandas as pd
import csv
import os
import re
import natsort
from PIL import Image
from matplotlib import pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import time


class ChristmasImages(Dataset):                    # Dataset class for reading the images

    # for bigger datasets it can be necessary to just store the path and load each element in getitem.

    def __init__(self, path, training):      #init function loads the data
        super().__init__()

        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier

        self.training = training
        self.path = path
        self.tensor_image = {}

        # Building training Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(256),
            transforms.RandomRotation(degrees = (-180,180)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

        # Building validation Data transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(192),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

        if self.training == True:
            self.dataset = ImageFolder(path + './dataset/train',transform = self.train_transform)
        else:
            self.path = path
            self.unsorted_image = os.listdir(self.path)
            self.sorted_image = natsort.natsorted(self.unsorted_image)

    def __len__(self):
        return len(self.dataset.samples)



    def __getitem__(self, index):                   #getitem function that returns a certain item given an id < length
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training == True:
            img = self.dataset[index][0]
            label = self.dataset[index][1]
            return (img,label )
        else:
            img_loc = os.path.join(self.path,
                                   self.sorted_image[index])
            # opening image using cv2 function
            #            image = cv2.imread(img_loc)

            # opening image with PIL package
            img = Image.open(img_loc).convert("RGB")
            img = self.val_transform(img)
            return (img, )
        raise NotImplementedError

path = r'C:\Users\vaibh\Desktop\Christmas Challenge'
batch_size = 32
train_dataset = ChristmasImages(path = path, training = True)

#Pytoch dataloader for training set
trainloader = DataLoader(train_dataset,batch_size = 32, shuffle = True,pin_memory=True)
train_batch = next(iter(trainloader))

train_images = train_batch[0]
train_labels = train_batch[1]

#Image Show of Training images
plt.figure(figsize=(12,10)) # specifying the overall grid size
for i in range(8):
    plt.subplot(2, 4, i+1)    # the number of images in the grid is 2*4 (8)
    plt.imshow(train_images[i], vmin=0., vmax=1.)
    plt.title('Label: {}'.format(train_labels[i]))


class ValidationSet(Dataset):

    def __init__(self, path):
        super().__init__()
        self.dataset = ChristmasImages(path + './dataset/val', training=False)

        with open(path + './dataset/val.csv') as file:
            reader = csv.reader(file)
            next(reader)
            labels = {}
            for row in reader:
                labels[int(row[0])] = int(row[1])
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.labels[idx]
        return image, label

val_batchsize = 1
loader = DataLoader(ValidationSet(path), batch_size = 1, shuffle = True,pin_memory=True)

#Iterating through the validation Dataloader
val_batch = next(iter(loader))
val_images = val_batch[0]
val_labels = val_batch[1]

plt.figure(figsize=(8,6)) # specifying the overall grid size
for i in range(val_batchsize):
    plt.subplot(1, val_batchsize, i+1)
    plt.imshow(val_images[i, 0], vmin=0., vmax=1.)
    plt.title('Label: {}'.format(val_labels[i]))