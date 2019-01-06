class DefaultConfigs(object):
    root_path = "/home/trinhnh1/Documents/kaggle/human-protein/input"
    train_data = root_path + "/train_all_new/" # where is your train data
    test_data = root_path + "/test_jpg/"   # your test data
    sample_submission = root_path + "/sample_submission.csv"
    train_kaggle_csv = root_path + "/train.csv"
    train_external_csv = root_path + "/external_data/img/train.csv"
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "resnet50_bcelog"
    seed = 2050
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.0001
    weight_decay = 0.000000001
    batch_size = 40
    epochs = 30
    thresold = 0.2
    n_tta = 5
    n_fold = 5

config = DefaultConfigs()

import os
import pandas as pd 
from tqdm import tqdm
import numpy as np
from PIL import Image

def read_images():
    # df1 = pd.read_csv(config.train_kaggle_csv)
    # df2 = pd.read_csv(config.train_external_csv)
    # all_files = pd.concat([df1, df2])
    all_files = pd.read_csv(config.train_external_csv)
    
    for index in range(len(all_files)):
        # print(index)
        row = all_files.iloc[index]
        filename = config.train_data + str(row.Id)
        print(filename)
        #use only rgb channels
        # if config.channels == 4:
        #     images = np.zeros(shape=(512,512,4))
        # else:
        #     images = np.zeros(shape=(512,512,3))
        r = np.array(Image.open(filename+"_red.jpg")) 
        g = np.array(Image.open(filename+"_green.jpg")) 
        b = np.array(Image.open(filename+"_blue.jpg")) 
        y = np.array(Image.open(filename+"_yellow.jpg")) 

if __name__ == '__main__':
    read_images()