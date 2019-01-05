import math
import torch
import pandas as pd
import numpy as np

from torch import nn
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# sacred import
from sacred import Ingredient
criterion_ingredient = Ingredient('criterion')


@criterion_ingredient.config
def cfg():
    loss   = 'logbce' # logbce / focal (default: logbce)
    weight = 'log'    # log / linear / none (default: log)


@criterion_ingredient.capture
def load_loss(label_count, loss):
    weight = get_weight(label_count)
    if loss == 'focal': return focal_loss
    else: return nn.BCEWithLogitsLoss(weight=weight).cuda()


@criterion_ingredient.capture
def get_weight(label_count, weight, mu=0.5):
    total = np.sum(list(label_count.values()))
    keys = label_count.keys()
    class_weight_linear = OrderedDict()
    class_weight_log = OrderedDict()
    class_weight_flat = OrderedDict()

    for key in keys:
        score = total / float(label_count[key])
        score_log = math.log(mu * total / float(label_count[key]))
        class_weight_linear[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key]    = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)
        class_weight_flat[key]   = 1

    if weight == 'log':
        class_weight = class_weight_log
    if weight == 'linear':
        class_weight = class_weight_linear
    else:
        class_weight = class_weight_flat

    return torch.FloatTensor(list(class_weight.values())).cuda()


# =========================
def cal_f1_scores(cfs_mats):
    f1_scores = []
    for cfs_mat in cfs_mats:
        (tn, fp, fn, tp) = cfs_mat
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return f1_scores

def update_macro_f1(output, target, cfs_mats, threshold, n_classes):
    preds = output.sigmoid().cpu() > threshold
    cfs_mats = [cfs_mats[i] + confusion_matrix(target[:, i], preds[:, i]).ravel()
                for i in range(n_classes)]
    f1_scores = cal_f1_scores(cfs_mats)
    return cfs_mats, f1_scores

def focal_loss(output, target, alpha=0.25, gamma=2):
    x, p, t = output, output.sigmoid(), target
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
