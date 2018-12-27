import os
import time
import json
import torch
import random
import warnings
import torchvision
import numpy as np
import pandas as pd

from utils import *
from data import HumanDataset
from tqdm import tqdm
from config import config
from datetime import datetime
from models.model import*
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from edafa import ClassPredictor
from torchvision import transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# 1. test model on public dataset and save the probability matrix
def test(test_loader, model, folds):
    sample_submission_df = pd.read_csv(config.sample_submission)
    #1.1 confirm the model converted to cuda
    filenames, labels, submissions= [], [], []
    model.cuda()
    model.eval()
    submit_results = []
    for i, (input, filepath) in enumerate(tqdm(test_loader)):
        #1.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()

            labels.append(label > config.thresold)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/{}_bestloss_submission.csv'.format(config.model_name), index=None)

# 2. main function
def main():
    fold = 0
    # 2.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)

    # 2.2 get model
    model = get_net()
    model.cuda()

    #print(all_files)
    test_files = pd.read_csv(config.sample_submission)

    test_gen = HumanDataset(test_files, config.test_data, augument=False, mode="test")
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

    best_model = torch.load("{}/{}_fold_{}_model_best_loss.pth.tar".format(config.best_models,config.model_name,str(fold)))
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test(test_loader, model, fold)

if __name__ == "__main__":
    main()
