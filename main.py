from data_loader.baseline_data_loader import BaselineModelDataLoader
from models.simple_mnist_model import SimpleMnistModel
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from preprocessing.preprocessing import ImagePreprocessor
from predicts.predict_generator import PredictGenerator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import numpy as np
import os
from sklearn.model_selection import RepeatedKFold
import pandas as pd

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    # list out train & test files
    train_path = os.path.join(config.data.root_path, "train")
    test_path = os.path.join(config.data.root_path, "test")
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    percentage = np.round(len(test_files) / len(train_files) * 100)
    print("The test set size turns out to be {} % compared to the train set.".format(percentage))

    # K-fold cross-validation
    splitter = RepeatedKFold(
        n_splits=config.data.kfold_cv.n_splits,
        n_repeats=config.data.kfold_cv.n_repeats,
        random_state=config.data.kfold_cv.random_state
    )

    labels = \
        pd.read_csv(os.path.join(config.data.root_path, "train.csv"))

    partitions = []
    for train_idx, test_idx in splitter.split(labels.index.values):
        partition = {}
        partition["train"] = labels.Id.values[train_idx]
        partition["validation"] = labels.Id.values[test_idx]
        partitions.append(partition)
        print("TRAIN: {0} TEST: {1}".format(train_idx, test_idx))
        print("TRAIN: {0} TEST: {1}".format(len(train_idx), len(test_idx)))

    # preprocesing
    preprocessor = ImagePreprocessor(config)

    # training the baseline model on the first cv-fold
    print('Create the data generator.')
    partition = partitions[0]
    training_generator = BaselineModelDataLoader(config, partition["train"], labels, preprocessor)
    validation_generator = BaselineModelDataLoader(config, partition["validation"], labels, preprocessor)
    predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)

    

    # print('Create the model.')
    # model = SimpleMnistModel(config)

    # print('Create the trainer')
    # trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)

    # print('Start training the model.')
    # trainer.train()


if __name__ == '__main__':
    main()
