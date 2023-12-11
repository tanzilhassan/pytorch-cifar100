""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class CustomCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.cifar100 = datasets.CIFAR100(
            root=root, train=train, transform=transform, target_transform=target_transform, download=True
        )
        
        # Define the list of classes to keep (80 classes out of 100)
        self.selected_classes = list(range(80))
        self.class_mapping = {class_idx: idx for idx, class_idx in enumerate(self.selected_classes)}
        
        # Filter the data and targets to include only the selected classes
        filtered_data = []
        filtered_targets = []
        
        for data, target in zip(self.cifar100.data, self.cifar100.targets):
            if target in self.selected_classes:
                filtered_data.append(data)
                filtered_targets.append(self.class_mapping[target])
        
        self.data = np.array(filtered_data)
        self.targets = np.array(filtered_targets)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image and apply transforms if provided
        img = ToTensor()(img)
        
        if self.cifar100.transform is not None:
            img = self.cifar100.transform(img)
        
        if self.cifar100.target_transform is not None:
            target = self.cifar100.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# Example usage
if __name__ == '__main__':
    custom_cifar100_train = CustomCIFAR100(root='./data', train=True, transform=ToTensor())
    custom_cifar100_test = CustomCIFAR100(root='./data', train=False, transform=ToTensor())

    print("Number of classes:", len(custom_cifar100_train.selected_classes))
    print("Number of training samples:", len(custom_cifar100_train))
    print("Number of test samples:", len(custom_cifar100_test))


