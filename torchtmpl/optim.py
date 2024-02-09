# coding: utf-8

# External imports
import torch
import torch.nn as nn

from . import utils

class Top30Loss(nn.Module):
    def __init__(self):
        super(Top30Loss, self).__init__()

    def forward(self, predicted, targets):
        indexs = utils.get_top_30(predicted)
        accuracy = [species in indexs[i] for i, species in enumerate(targets)]
        return accuracy.count(False) / targets.size(0)

def get_loss(lossname):
    if lossname == "top30loss":
        return Top30Loss()
    return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
