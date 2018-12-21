import os 
import time
import gc 
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
from models.model import *
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from timeit import default_timer as timer
from sklearn.metrics import f1_score

# 1. set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# create log folder
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
    
log = Logger()
log.open("./logs/%s_log_train.txt"%config.model_name, mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

# train and valid model on each epoch
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, valid_loss, best_results, start):
    for epoch in range(num_epochs):
        best_model = model
        best_acc = 0.0
        
        # each epoch has training and validation phase
        for phase in ['train', 'val']:
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
                print('-'*10)
                
            losses = AverageMeter()
            f1 = AverageMeter()
            
            if phase == 'train':
                mode = 'train'
                is_train = True
                data_loader = train_loader
                optimizer = lr_scheduler(optimizer, epoch)
                model.train() # set model to training mode
            else:
                mode = 'val'
                is_train = False
                data_loader = val_loader
                model.eval()  # set model to validation mode
                
            counter = 0
            # Iterate over data
            with torch.set_grad_enabled(is_train):
                for (i, inputs, labels) in enumerate(data_loader):
                    inputs = inputs.cuda(non_blocking=True)
                    labels = torch.from_numpy(np.array(labels)).float().cuda(non_blocking=True)
                    print(inputs.size())
                    
                    # compute outputs
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses.update(loss.item(),inputs.size(0))
                    f1_batch = f1_score(labels, outputs.sigmoid().cpu() > 0.15, average='macro')
                    f1.update(f1_batch, inputs.size(0))
                    
                    # set gradient to zero to delete history of computations in previous epoch. 
                    # track operations so that differentiation can be done automatically.
                    optimizer.zero_grad()
                    counter += 1

                    # backward and optimize only if in training phase
                    if phase == 'train':
                        #print('loss backward')
                        loss.backward()
                        #print('done loss backward')
                        optimizer.step()
                    else:
                        pass
    
    return [losses.avg, f1.avg]


# 2. test model on public dataset and save the probability matrix 
def test(test_loader, model, folds):
    sample_submission_df = pd.read_csv("/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/sample_submission.csv")
    #2.1 confirm the model converted to cuda
    filenames, labels, submissions = [], [], []
    model.cuda()
    model.eval()
    submit_results = []
    for i, (input, filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            #print(label > 0.5)
           
            labels.append(label > 0.15)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestloss_submission.csv'%config.model_name, index=None)


# 3. main function
def main():
    # 3.1 make folder
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    kfold = 5
    start_epoch = 0
    best_loss = 999
    best_f1 = 0
    best_results = [np.inf, 0]
    val_metrics = [np.inf, 0]
    resume = False
    start = timer()

    # get train
    # train data, this data include external data
    df1 = pd.read_csv("/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/train.csv")
    df2 = pd.read_csv("/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/external_data/img/train.csv")
    all_files = pd.concat([df1, df2])
    
    # get test data, this use for making submission file
    test_files = pd.read_csv("/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/sample_submission.csv")   
    
    # class weight. This is imbalanced data problem, so we will add class weight to loss function
    weights = cls_wts(name_label_dict, 0.3)[1]
    class_weights = torch.FloatTensor(weights).cuda()

    # get model
    model = get_net()
    model.cuda()
    
    # use kfold cross validation to find out best model. 
    # because this problem has imbalanced data, we use advanced split method. Scikitlearn is not strong enough.
    msss = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=config.seed)
    all_files_org = all_files.copy()
    all_files_org['Target'] = all_files_org.apply(lambda x: x['Target'].split(' '), axis=1)
    X = all_files_org['Id'].tolist()
    y = all_files_org['Target'].tolist()
    y = MultiLabelBinarizer().fit_transform(y)
    
    for train_data_list, val_data_list in msss.split(X, y):
        print("Train data: {}. Valid data: {}".format(train_data_list, val_data_list))
        # load dataset
        # train data
        train_gen = HumanDataset(train_data_list, config.train_data, mode="train")
        train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        # val data
        val_gen = HumanDataset(val_data_list, config.train_data, augument=False,mode="train")
        val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        # test data
        test_gen = HumanDataset(test_files, config.test_data, augument=False, mode="test")
        test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

        # freeze all layers
        
    
    
    

if __name__ == '__main__':
    main()