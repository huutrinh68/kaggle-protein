import os

# as usual 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# cross-validation split
from sklearn.utils import class_weight, shuffle

# image preprocessing
import PIL
from PIL import Image
import cv2
from imgaug import augmenters as iaa

# keras modules
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras.models import 
import keras.losses
# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# processbar and datetime
from tqdm import tqdm
import datetime as dt

# slack nofify
from slackclient import SlackClient

# limit GPU when training data
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.Session(config=config)
K.set_session(sess)

# warning ignore
import warnings
warnings.filterwarnings("ignore")

root_path = "/home/trinhnh1/Documents/train_data/kaggle/human-protein/input"
start = dt.datetime.now()