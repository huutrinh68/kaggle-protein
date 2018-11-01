"""
    import common library
"""
import os
import ast
import datetime as dt
import argparse

# slack nofify
from slackclient import SlackClient

# matplot and graph
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns

# as usual
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

# image processing and augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import albumentations
import cv2

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,
    Resize, Normalize, Rotate, RandomCrop, Crop, CenterCrop, DualTransform
)

# tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import optimizers
from keras.callbacks import LambdaCallback

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

###################################
# limit GPU when training data
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.Session(config=config)
K.set_session(sess)
###################################

root_path = '/home/trinhnh1/Documents/train_data/kaggle/human_protein/'

start = dt.datetime.now()
