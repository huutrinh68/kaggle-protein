from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F

# def get_net():
#     model = bninception(pretrained="imagenet")
#     model.global_pool = nn.AdaptiveAvgPool2d(1)
#     model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
#     model.last_linear = nn.Sequential(
#                 nn.BatchNorm1d(1024),
#                 nn.Dropout(0.5),
#                 nn.Linear(1024, config.num_classes),
#             )
#     return model


def get_net():
    model = bninception(pretrained="imagenet")
    new_features = nn.Sequential(*list(model.children()))
    # get the pre-trained weights of the first layer
    pretrained_weights = new_features[0].weight

    new_features[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    # For 4-channel weight should randomly initialized with Gaussian
    # new_features[0].weight.data.normal_(0, 0.001)
    new_features[0].weight.data[:, 1:4, :, :] = pretrained_weights
    # For RGB it should be copied from pretrained weights
    new_features[0].weight.data[:, :3, :, :] = pretrained_weights

    # print(new_features[0].weight)
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = new_features[0]
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 28),
            )
    return model