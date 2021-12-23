# data.py

import os
import random
import glob
import cv2
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import *


if torchvision_dataset:
    # for torchvision dataset like mnist etc.
    
    # data transform
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    # download mnist data
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # dataloader
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)

    # check 
    batch = next(iter(train_loader))
    images, labels = batch

    # combine train and test loader into dataloader 
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['test'] = test_loader

    # combine train and test sizes into dataset_sizes
    dataset_sizes = {}
    dataset_sizes['train'] = len(mnist_trainset)
    dataset_sizes['test'] = len(mnist_testset)
    
else:
    # custom dataset 

    class MyDataset(Dataset):
        def __init__(self, data):
            self.images = data['images']
            self.labels = data['labels']

        def __getitem__(self,index):
            x = torch.tensor(self.images[index])
            x = torch.transpose(torch.transpose(x, 2,0), 1,2)
            # x = rearrange(x, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
            y = torch.tensor(self.labels[index])
            return {0:x, 1:y}

        def __len__(self):
            return len(self.images)

    # define classes
    classes = os.listdir(images_folder)
    class_dict = dict(zip(classes, range(0,len(classes))))
  
    images = []
    labels = []
    for c in classes:
        images_path = os.path.join(images_folder, c)
        imgs = [cv2.resize(cv2.imread(i), (256,256)) for i in glob.glob(images_path + '/*')]
        images.extend(imgs)
        label = class_dict[c]
        labels.extend([label]*len(images))

    # shuffle classes 
    
    shuf = list(zip(images, labels))
    random.shuffle(shuf)
    images, labels = zip(*shuf)
    dataset = {}
    dataset['images'] = images
    dataset['labels'] = labels

    len_dataset = len(dataset['images'])
    idx = int(train_test_split * len_dataset)
    train_idx = batch_size * (idx // batch_size)
    test_idx = train_idx + batch_size * ((len_dataset - train_idx) // batch_size)

    train_dataset = {'images':dataset['images'][0:train_idx], 'labels':dataset['labels'][0:train_idx]}
    test_dataset = {'images':dataset['images'][train_idx:test_idx], 'labels':dataset['labels'][train_idx:test_idx]}

    # dataloader
    train_loader = DataLoader(MyDataset(data=train_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(data=test_dataset), batch_size=batch_size, shuffle=True)

    # check 
    batch = next(iter(train_loader))
    images, labels = batch

    # combine train and test loader into dataloader 
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['test'] = test_loader

    # combine train and test sizes into dataset_sizes
    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset['images'])
    dataset_sizes['test'] = len(test_dataset['images'])
