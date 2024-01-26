# coding: utf-8

# Standard imports

# External imports
import torch
import torch.nn as nn

"""
Task:
un modèle convolutif "fait maison" avec des blocs de la forme :  [2 x [Conv3x3 - Batch Norm - Relu ]] - MaxPool2x2 ,
en terminant le réseau par un global average pooling et une ou des couches linéaires/activation pour produire au final les scores des 10 classes
"""

class BiModel(nn.Module)
    def __init__(self, )

def conv3x3(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    ]
    

def CNN_Custom(cfg, input_size, num_classes):
    layers = []
    cin = input_size[0]
    cout = 16
    #num layers should be 2
    layers.extend(conv3x3(cin, cout))
    layers.extend(conv3x3(cout, 2*cout))
    conv_model = nn.Sequential(*layers)

# size after average pooling 256x5408
# (yes i cheated to calculate the general output of the conv+maxpool+avg, but its approximatly (16*2) * (126/3/3)^2 in this case because our image is 128)
    return nn.Sequential(
        conv_model,
        nn.MaxPool2d(kernel_size=3),
        nn.AvgPool2d(kernel_size=3),
        nn.Flatten(start_dim=1),
        nn.Linear(5408, num_classes),
        nn.Dropout(0.2)
        )
