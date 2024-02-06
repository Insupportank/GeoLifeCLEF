# coding: utf-8

# Standard imports

# External imports
import torch
import torch.nn as nn
from torchvision import models
from functools import reduce
import operator


class FeaturesMLP(nn.Module):
    def __init__(self, features_input_size, features_output_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(features_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_output_size)
            )
        
    def forward(self, x):
        return self.seq(x)
    
class CNN(nn.Module):
    def __init__(self, cfg, image_input_size, output_size):
        super().__init__()
        def conv_relu_bn(cin, cout):
            return [
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(cout),
            ]
        def conv_down(cin, cout):
            return [
                nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(cout),
            ]
        
        layers = []
        cin = image_input_size[0]
        cout = 16
        for i in range(cfg["num_layers"]):
            layers.extend(conv_relu_bn(cin, cout))
            layers.extend(conv_relu_bn(cout, cout))
            layers.extend(conv_down(cout, 2 * cout))
            cin = 2 * cout
            cout = 2 * cout
        conv_model = nn.Sequential(*layers)

        # Compute the output size of the convolutional part
        probing_tensor = torch.zeros((1,) + image_input_size)
        out_cnn = conv_model(probing_tensor)  # B, K, H, W
        num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
        out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, output_size)]
        self.seq = nn.Sequential(conv_model, *out_layers)
           
    def forward(self, x):
        return self.seq(x)

class MyResNet(nn.Module):
    def __init__(self, output_size):
        super(MyResNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        modules = list(resnet18.children())[:-2]
        self.resnet18 = nn.Sequential(*modules)
        
        num_features = resnet18.fc.in_features
        self.custom_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 2*output_size),
            nn.ReLU(),
            nn.Linear(2*output_size, output_size)
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.custom_layers(x)
        return x
# multiple models https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383
# eval() and train() do work recurcively !! :)
# gradient also does include child models. (except with .requires_grad=False)

class BiModel(nn.Module):
    def __init__(self, cfg, input_sizes, num_classes):
        super().__init__()
        # output sizes of cnn and features mlp can be chosen randomly, but by putting them at the number of classes / 4 makes both models try to extract inportant info withough behind to big
        image_input_size, features_input_size = input_sizes
        cnn_output_size = num_classes // 8
        features_output_size = num_classes // 8
        self.image_model = MyResNet(cnn_output_size)
        self.features_model = FeaturesMLP(features_input_size, features_output_size)
        self.seq = nn.Sequential(
            nn.Linear(cnn_output_size + features_output_size, num_classes//2), # 1226x2455
            nn.ReLU(),
            nn.Linear(num_classes//2, num_classes)
            )

    def forward(self, x):
        image, features = x
        cnn_output = self.image_model(image)
        mlp_output = self.features_model(features)

        # print(image.size()) # torch.Size([32, 3, 256, 256])
        # print(features.size()) # torch.Size([32, 29])

        concat = torch.cat((cnn_output, mlp_output), dim=1)
        # print(concat.size()) # torch.Size([32, 1226])
        return self.seq(concat)