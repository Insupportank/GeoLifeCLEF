# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from . import data_set

def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    base_dataset = data_set.GeoLifeDataset(data_config["trainpath"], country=data_config["data_language"], data_portion=data_config["data_portion"])
    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = max(base_dataset.categories) + 1
    # print(base_dataset[0][0]["image"].shape) #torch.Size([3, 256, 256])
    # print(base_dataset[0][0]["features"].shape) #torch.Size([29])
    input_sizes = (tuple(base_dataset[0][0]["image"].shape), tuple(base_dataset[0][0]["features"].shape)[0])

    return train_loader, valid_loader, input_sizes, num_classes


def get_test_dataloader(data_config, use_cuda):
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    base_dataset = data_set.GeoLifeDataset(data_config["trainpath"], file_type="test", country="all", transform=None, data_portion=1.)
    logging.info(f"  - I loaded {len(base_dataset)} samples from test set")

    # Build the dataloader
    test_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = 17037 # Find way to have this number withough using base_dataset.categories that uses the number of different spicies_id.
    # print(base_dataset[0][0]["image"].shape) #torch.Size([3, 256, 256])
    # print(base_dataset[0][0]["features"].shape) #torch.Size([29])
    input_sizes = (tuple(base_dataset[0][0]["image"].shape), tuple(base_dataset[0][0]["features"].shape)[0])

    return test_loader, input_sizes, num_classes
