import os
import sys 
import json
import torch
import shutil
import numpy as np 
import math
from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable
# save best model
def save_checkpoint(state, is_best_loss,is_best_f1,fold):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    if is_best_f1:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))

# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25,gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        t = Variable(y).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError

# Class abundance for protein dataset
labels_dict = {
    0: 12885,
    1: 1254,
    2: 3621,
    3: 1561,
    4: 1858,
    5: 2513,
    6: 1008,
    7: 2822,
    8: 53,
    9: 45,
    10: 28,
    11: 1093,
    12: 688,
    13: 537,
    14: 1066,
    15: 21,
    16: 530,
    17: 210,
    18: 902,
    19: 1482,
    20: 172,
    21: 3777,
    22: 802,
    23: 2965,
    24: 322,
    25: 8228,
    26: 328,
    27: 11
}

def class_weight(labels_dict, mu=0.5):
    values = list(labels_dict.values())
    keys = list(labels_dict.keys())
    total = np.sum(values)
    class_weight = {}
    class_weight_log = {}
    print('\ntoal:')
    print(total)
    print('\nvalues:')  
    print(values)

    for key in keys:
        score = total / float(labels_dict[key])
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    return class_weight, class_weight_log


name_label_dict = {
    0:   ('Nucleoplasm', 12885),
    1:   ('Nuclear membrane', 1254),
    2:   ('Nucleoli', 3621),
    3:   ('Nucleoli fibrillar center', 1561),
    4:   ('Nuclear speckles', 1858),
    5:   ('Nuclear bodies', 2513),
    6:   ('Endoplasmic reticulum', 1008),   
    7:   ('Golgi apparatus', 2822),
    8:   ('Peroxisomes', 53), 
    9:   ('Endosomes', 45),
    10:  ('Lysosomes', 28),
    11:  ('Intermediate filaments', 1093), 
    12:  ('Actin filaments', 688),
    13:  ('Focal adhesion sites', 537),  
    14:  ('Microtubules', 1066), 
    15:  ('Microtubule ends', 21),
    16:  ('Cytokinetic bridge', 530),
    17:  ('Mitotic spindle', 210),
    18:  ('Microtubule organizing center', 902),
    19:  ('Centrosome', 1482),
    20:  ('Lipid droplets', 172),
    21:  ('Plasma membrane', 3777),
    22:  ('Cell junctions', 802),
    23:  ('Mitochondria', 2965),
    24:  ('Aggresome', 322),
    25:  ('Cytosol', 8228),
    26:  ('Cytoplasmic bodies', 328),   
    27:  ('Rods &amp; rings', 11)
    }

n_labels = 50782

def cls_wts(label_dict, mu=0.5):
    prob_dict, prob_dict_bal = [], []
    max_ent_wt = 1/28
    for i in range(28):
        prob = label_dict[i][1]/n_labels
        prob_dict.append(prob)
        if prob > max_ent_wt:
            prob_dict_bal.append(prob-mu*(prob - max_ent_wt))
        else:
            prob_dict_bal.append(prob+mu*(max_ent_wt - prob))         
    return prob_dict, prob_dict_bal

# if __name__=='__main__':
#     print('\nTrue class weights:')
#     print(class_weight(labels_dict)[0])
#     print('\nLog-dampened class weights:')
#     print(class_weight(labels_dict)[1])
#     print('\nprob_dict:')
#     print(cls_wts(name_label_dict)[0])
#     print('\nprob_dict_bal:')
#     print(cls_wts(name_label_dict, 0.3)[1])