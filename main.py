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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

from tensorboardX import SummaryWriter
import gc

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    cfs_mats = [np.zeros(4) for i in range(config.num_classes)] # confusion matrix for each class
    macro_f1 = None
    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))

        # update tn, fp, fn, tp
        preds = output.sigmoid().cpu() > config.thresold
        cfs_mats = [cfs_mats[i] + confusion_matrix(target[:, i], preds[:, i]).ravel()
                    for i in range(config.num_classes)]
        f1_scores = cal_f1_scores(cfs_mats)
        macro_f1 = np.nanmean(f1_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, macro_f1,
                valid_loss[0], valid_loss[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)

        # Tensorboard
        if i % 50 == 49:
            f1_scores = cal_f1_scores(cfs_mats)
            f1_scores_dict = {'class_'+str(i):f1_scores[i] for i in range(config.num_classes)}
            writer.add_scalar(config.model_name + '/data/train_loss', loss, batch_global_counter)
            writer.add_scalar(config.model_name + '/data/train_f1', macro_f1, batch_global_counter)
            writer.add_scalars(config.model_name + '/data/class_train_f1', f1_scores_dict, batch_global_counter)
            batch_global_counter += 1

    log.write("\n")
    return [losses.avg, macro_f1]

# evaluate function
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    cfs_mats = [np.zeros(4) for i in range(config.num_classes)] # confusion matrix for each class
    macro_f1 = None
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

            # update tn, fp, fn, tp
            preds = output.sigmoid().cpu() > config.thresold
            cfs_mats = [cfs_mats[i] + confusion_matrix(target[:, i], preds[:, i]).ravel()
                        for i in range(config.num_classes)]
            f1_scores = cal_f1_scores(cfs_mats)
            macro_f1 = np.nanmean(f1_scores)

            # print
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,
                    train_loss[0], train_loss[1],
                    losses.avg, macro_f1,
                    str(best_results[0])[:8], str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")

        writer.add_scalar(config.model_name + '/data/eval_loss', losses.avg, epoch)
        writer.add_scalar(config.model_name + '/data/eval_f1', macro_f1, epoch)
    return [losses.avg, macro_f1]

# main function
def main():
    n_fold = config.n_fold
    # mkdirs folder
    for fold in range(config.n_fold):
        fold_path = config.weights + config.model_name + os.sep + str(fold)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    
    start = timer()

    # initial
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
    del df1, df2
    gc.collect()
    
    # Stratified train test split
    labels = [np.array(list(map(int, str_label.split(' '))))
              for str_label in all_files.Target]
    y  = [np.eye(config.num_classes,dtype=np.float)[label].sum(axis=0)
          for label in labels]
    mskf = MultilabelStratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)

    # train with kfold
    preds = []
    for fold, (train_index, val_index) in enumerate(mskf.split(all_files, y)):
        # get model
        model = get_net()
        model.cuda()
        
        train_data_index, val_data_index = train_index, val_index
        train_data_list = all_files.iloc[train_data_index].reset_index()
        val_data_list = all_files.iloc[val_data_index].reset_index()

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

        del train_df_orig
        gc.collect()

        # compute class weight
        target = train_data_list.apply(lambda x: x['Target'].split(' '), axis=1)
        y = target.tolist()
        list_order = [str(i) for i in range(28)]
        mlb = MultiLabelBinarizer(list_order)
        y = mlb.fit_transform(y)
        labels_dict = dict()
        count_classes = np.sum(y, axis=0)
        for i,count in enumerate(count_classes):
            labels_dict[i] = count

        dampened_cw = create_class_weight(labels_dict)[1]
        tmp = list(dampened_cw.values())
        class_weight = torch.FloatTensor(tmp).cuda()

        del target, y
        gc.collect()

        # criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.BCEWithLogitsLoss(weight=class_weight).cuda()

        # load train dataset
        train_gen = HumanDataset(train_data_list, config.train_data, mode="train")
        train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        # load valid dataset
        val_gen = HumanDataset(val_data_list, config.train_data, augument=False, mode="train")
        val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # load test dataset
        test_gen = HumanDataset(test_files, config.test_data, augument=False, mode="test", tta=config.n_tta)
        test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

        # train and valid
        for epoch in range(0, config.epochs):
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

        # test with public dataset
        preds.append(test(test_loader, model, fold))
    
    # compute test predict over all fold
    test_pred = np.mean(preds, axis=0)

    # make submit file
    makesubmission(test_pred, test_loader)


def test(test_loader, model, fold):
    # evaluation model
    model.eval()

    test_pred = []
    for i, (inputs, _) in enumerate(tqdm(test_loader)):
        probs = []
        for input in inputs:
            with torch.no_grad():
                image_var = input.cuda(non_blocking=True)
                y_pred = model(image_var)
                prob = y_pred.sigmoid().cpu().data.numpy()
                probs.append(prob)
        test_pred.append(np.vstack(probs).mean(axis=0))

    return test_pred

def makesubmission(test_pred, test_loader):
    sample_submission_df = pd.read_csv(config.sample_submission)
    submissions= []
    pred = test_pred > config.thresold
    if len(pred) == 0: pred = [np.argmax(pred, axis=1)]

    for i, (_, _) in enumerate(tqdm(test_loader)):
        subrow = ' '.join(list([str(i) for i in np.nonzero(pred)[0]]))
        if len(subrow) == 0:
            subrow = np.argmax(test_pred)
        submissions.append(subrow)
    
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/{}_bestloss_submission.csv'.format(config.model_name), index=None)


if __name__ == "__main__":
    main()