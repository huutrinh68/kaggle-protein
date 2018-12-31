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
def test(test_loader, model, folds, ill_pairs):
    sample_submission_df = pd.read_csv(config.sample_submission)
    #1.1 confirm the model converted to cuda
    filenames, labels, submissions= [], [], []
    model.cuda()
    model.eval()
    submit_results = []
    for i, (inputs, _) in enumerate(tqdm(test_loader)):
        #1.2 change everything to cuda and get only basename
        probs = []
        for input in inputs:
            with torch.no_grad():
                image_var = input.cuda(non_blocking=True)
                y_pred = model(image_var)
                prob = y_pred.sigmoid().cpu().data.numpy()
                probs.append(prob)
        probs_agg = np.vstack(probs).mean(axis = 0)
        preds = probs_agg > config.thresold
        preds = np.nonzero(preds)[0]
        if len(preds) == 0:
            preds = [np.argmax(probs_agg)]
        preds = remove_ill_pairs(ill_pairs, preds, probs_agg)
        subrow = ' '.join(list([str(i) for i in preds]))
        if len(subrow) == 0:
            subrow = np.argmax(probs_agg)
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

    test_gen = HumanDataset(test_files, config.test_data, augument=False, mode="test", tta=config.n_tta)
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

    best_model = torch.load("{}/{}_fold_{}_model_best_loss.pth.tar".format(config.best_models,config.model_name,str(fold)))
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])

    ill_pairs = get_ill_pairs()
    test(test_loader, model, fold, ill_pairs)

if __name__ == "__main__":
    main()