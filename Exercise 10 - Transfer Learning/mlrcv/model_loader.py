import torch
from torchvision import models
import typing


import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, BatchNorm1d, ReLU, Flatten

def build_classifier(num_class: int) -> torch.nn.modules.container.Sequential:
    """
    This function builds the classifier to classify the resnet output features:

    Args:
        - num_class (int): number of class on the data to be defined as the classifier output features size

    Returns:
        - classifier (torch.nn.modules.container.Sequential): the classifier model to classify the resnet output features
    """

    classifier = torch.nn.Sequential(
       Linear(512, num_class)
        # add torch layers to build your classifier
    )

    return classifier