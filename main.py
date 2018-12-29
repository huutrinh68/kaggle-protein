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
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from tensorboardX import SummaryWriter
import gc
# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

writer = SummaryWriter()

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

batch_global_counter = 0
def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start):
    global batch_global_counter
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))

        f1_batch = f1_score(target,output.sigmoid().cpu() > config.thresold, average='macro')
        f1.update(f1_batch,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg,
                valid_loss[0], valid_loss[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)

        # Tensorboard
        if i % 10 == 9:
            writer.add_scalar('data/train_loss', loss, batch_global_counter)
            writer.add_scalar('data/train_f1', f1_batch, batch_global_counter)
            batch_global_counter += 1

    log.write("\n")
    return [losses.avg,f1.avg]

# 2. evaluate fuunction
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            #image_var = Variable(images).cuda()
            #target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(images_var)
            loss = criterion(output,target)
            losses.update(loss.item(), images_var.size(0))
            f1_batch = f1_score(target, output.sigmoid().cpu().data.numpy() > config.thresold, average='macro')
            f1.update(f1_batch, images_var.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,
                    train_loss[0], train_loss[1],
                    losses.avg, f1.avg,
                    str(best_results[0])[:8], str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")

        writer.add_scalar('data/eval_loss', losses.avg, epoch)
        writer.add_scalar('data/eval_f1', f1.avg, epoch)
    return [losses.avg, f1.avg]

# 4. main function
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    # 4.2 get model
    model = get_net()
    model.cuda()
    # load old weight trained model
    #model.load_state_dict(torch.load("{}/{}_fold_{}_model_best_loss.pth.tar".format(config.best_models,config.model_name,str(fold)))["state_dict"])

    start_epoch = 0
    best_loss = 999
    best_f1 = 0
    best_results = [np.inf, 0]
    val_metrics = [np.inf, 0]
    resume = False
    # get train
    # train data, this data include external data
    df1 = pd.read_csv(config.train_kaggle_csv)
    df2 = pd.read_csv(config.train_external_csv)
    all_files = pd.concat([df1, df2])
    test_files = pd.read_csv(config.sample_submission)

    # Stratified train test split
    labels = [np.array(list(map(int, str_label.split(' '))))
              for str_label in all_files.Target]
    y  = [np.eye(config.num_classes,dtype=np.float)[label].sum(axis=0)
          for label in labels]
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.13, random_state=2050)
    for train_index, val_index in msss.split(all_files, y): # Only run one time
        train_data_index, val_data_index = train_index, val_index
    train_data_list = all_files.iloc[train_data_index].reset_index()
    val_data_list = all_files.iloc[val_data_index].reset_index()
    # train_data_list, val_data_list = train_test_split(all_files, test_size = 0.13, random_state = 2050)

    # create duplicate for low data
    # https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/74374#437548
    train_df_orig = train_data_list.copy()
    lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
    for i in lows:
        target = str(i)
        indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
        train_data_list = pd.concat([train_data_list, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target+" ")].index
        train_data_list = pd.concat([train_data_list, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" "+target)].index
        train_data_list = pd.concat([train_data_list, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" "+target+" ")].index
        train_data_list = pd.concat([train_data_list, train_df_orig.loc[indicies]], ignore_index=True)

    del df1, df2, train_df_orig
    gc.collect()

    # compute class weight
    target = train_data_list.apply(lambda x: x['Target'].split(' '), axis=1)
    y = target.tolist()
    list_oder = [str(i) for i in range(28)]
    mlb = MultiLabelBinarizer(list_order)
    y = mlb.fit_transform(y)
    labels_dict = dict()
    count_classes = np.sum(y, axis=0)
    for i,count in enumerate(count_classes):
        labels_dict[i] = count

    del target, y
    gc.collect()

    dampened_cw = create_class_weight(labels_dict)[1]
    tmp = list(dampened_cw.values())
    class_weight = torch.FloatTensor(tmp).cuda()

    # criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(weight=class_weight).cuda()

    # load dataset
    train_gen = HumanDataset(train_data_list, config.train_data, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    val_gen = HumanDataset(val_data_list, config.train_data, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    start = timer()

    #train
    for epoch in range(0,config.epochs):
        scheduler.step(epoch)
        # train
        lr = get_learning_rate(optimizer)
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
        # val
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)
        # check results
        is_best_loss = val_metrics[0] < best_results[0]
        best_results[0] = min(val_metrics[0], best_results[0])
        is_best_f1 = val_metrics[1] > best_results[1]
        best_results[1] = max(val_metrics[1], best_results[1])
        # save model
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_loss":best_results[0],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[1],
        }, is_best_loss, is_best_f1, fold)
        # print logs
        print('\r',end='', flush=True)
        log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "best", epoch, epoch,
                train_metrics[0], train_metrics[1],
                val_metrics[0], val_metrics[1],
                str(best_results[0])[:8], str(best_results[1])[:8],
                time_to_str((timer() - start), 'min'))
            )
        log.write("\n")
        time.sleep(0.01)

    best_model = torch.load("{}/{}_fold_{}_model_best_loss.pth.tar".format(config.best_models, config.model_name, str(fold)))
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])

if __name__ == "__main__":
    main()
