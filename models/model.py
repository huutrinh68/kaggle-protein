from torchvision import models
from pretrainedmodels.models import resnet34, resnet50, resnet152, bninception
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
import torch


def get_net():
    model = resnet50(pretrained="imagenet")

    # get the pre-trained weights
    pretrained_weights = list(model.parameters())
    
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((pretrained_weights[0], pretrained_weights[0][:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(204800),
                nn.Dropout(0.5),
                nn.Linear(204800, config.num_classes),
            )
    
    return model
