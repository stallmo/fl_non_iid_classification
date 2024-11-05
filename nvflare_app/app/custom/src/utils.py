import logging
import os

import torchvision
import torchvision.transforms as transforms
import torch

# make sure the import works regardless from where the code is called
from .le_net_cifar10 import LeNetCifar10
import torch.nn as nn


def get_model(name: str) -> nn.Module:
    """
    Get the model by name.
    :param name: Name of the data the model is applied to.
    :return: Pytorch model.
    """
    if name == 'CIFAR10':
        return LeNetCifar10()
    else:
        raise ValueError(f"Model {name} not found")


def get_loss(name: str):
    """
    Get the loss function by name.
    :param name: Name of the loss function.
    :return: Pytorch loss function.
    """
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {name} not found")


def get_data_centralized(dataset_name: str, batch_size: int, root_dir: str, val_size: float = 0.1,
                         apply_transforms: bool = True) -> tuple:
    """
    Get the data for centralized training.
    :param dataset_name: Name of the dataset.
    :param batch_size: Batch size.
    :param root_dir: Where to store the dataset.
    :param val_size: Size of the validation set. Must be between 0 and 1.
    :param apply_transforms: Specify if the data should be transformed.
        Currently only normalization is applied if True.
    :return: Tuple with train, validation and test data.
    """
    if dataset_name == 'CIFAR10':

        all_transforms = [transforms.ToTensor()]

        if apply_transforms:
            # values calculated from the training set during data exploration
            all_transforms.append(transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                       (0.24703223, 0.24348513, 0.26158784)))

            all_train_transformations = all_transforms + [transforms.RandomHorizontalFlip(),
                                                          # randomly flip the image horizontally
                                                          transforms.RandomCrop(32, padding=4),
                                                          # randomly crop the image
                                                          transforms.RandomRotation(15),  # rotate up to 15 degrees
                                                          ]
        transform = transforms.Compose(all_transforms)
        if apply_transforms:
            train_transformations = transforms.Compose(all_train_transformations)
        else:
            train_transformations = transform

        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=train_transformations)

        trainset, valset = torch.utils.data.random_split(trainset, [1 - val_size, val_size])

        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                               download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    return trainloader, valloader, testloader


def save_pt_model(model, path):
    """
    Save a Pytorch model.
    :param model: Pytorch model.
    :param path: Path to save the model.
    """
    # check if the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved at {path}")
