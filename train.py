import random
import torch
import warnings
import os
import numpy as np
import pandas as pd
from skeleton.trainer import Trainer
from sklearn.metrics import confusion_matrix
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.commands import print_config

# local import
from optimizer import optimizer_ingredient, load_optimizer
from criterion import criterion_ingredient, load_loss, cal_f1_scores
from model import model_ingredient, load_model
from data import data_ingredient, create_loader
from path import path_ingredient, prepair_dir

ex = Experiment('Sample', ingredients=[model_ingredient,      # model
                                       optimizer_ingredient,  # optimizer
                                       data_ingredient,       # data
                                       path_ingredient,       # path
                                       criterion_ingredient]) # criterion
ex.observers.append(MongoObserver.create(db_name='human_protein'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    max_epochs = 20
    threshold = 0.2
    resume = True
    debug = False
    if debug == True:
        max_epochs = 3


@ex.capture
def init(_run, seed, path):
    prepair_dir()
    ex.observers.append(FileStorageObserver.create(path['root'] + path['exp_logs'] + 'experiments/'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

    print('='*50)
    print_config(_run)
    print('='*50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using', torch.cuda.device_count(), 'gpus!')

    return device


@ex.capture
def after_init(trainer):
    trainer.cache = dict()


@ex.main
def main(_log, max_epochs, resume, model, optimizer, data, path, seed, debug, criterion):
    device = init()
    for fold in range(data['n_fold']):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(load_model()).to(device)
        else:
            model = load_model().to(device)
        optimizer = load_optimizer(model.parameters())
        train_loader, val_loader, label_count = create_loader(n_fold = data['n_fold'],
                                                              fold=fold,
                                                              seed=seed,
                                                              debug=debug)
        loss_func = load_loss(label_count)
        trainer = Trainer(
            alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
            code = get_dir_name(),
            fold=fold,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_func=loss_func,
            max_epochs=max_epochs,
            resume=resume,
            hooks={'after_init': after_init,
                   'after_train_iteration_end': after_train_iteration_end,
                   'after_val_iteration_end': after_val_iteration_end,
                   'before_epoch_start': before_epoch_start,
                   'after_epoch_end': after_epoch_end},
            additional_metrics = {'macro_f1'}
        )

        # trainer.train()
        try:
            trainer.train()
        except Exception as e:
            _log.error('Unexpected exception! %s', e)


@ex.capture
def before_epoch_start(trainer, data):
    trainer.cache['train_cfs_mats']  = [np.zeros(4) for i in range(data['n_classes'])]
    trainer.cache['train_f1_scores'] = None
    trainer.cache['train_macro_f1']  = 0
    trainer.cache['val_cfs_mats']    = [np.zeros(4) for i in range(data['n_classes'])]
    trainer.cache['val_f1_scores']   = None
    trainer.cache['val_macro_f1']    = 0

@ex.capture
def after_train_iteration_end(trainer, data, threshold):
    cfs_mats, f1_scores = update_macro_f1(trainer.cache['output'], trainer.cache['target'],
                                          trainer.cache['train_cfs_mats'], threshold, data.n_classes)
    trainer.cache['train_cfs_mats']  = cfs_mats
    trainer.cache['train_f1_scores'] = f1_scores
    trainer.cache['train_macro_f1']  = np.nanmean(f1_scores)


@ex.capture
def after_val_iteration_end(trainer, data, threshold):
    cfs_mats, f1_scores = update_macro_f1(trainer.cache['output'], trainer.cache['target'],
                                          trainer.cache['val_cfs_mats'], threshold, data.n_classes)
    trainer.cache['val_cfs_mats']  = cfs_mats
    trainer.cache['val_f1_scores'] = f1_scores
    trainer.cache['val_macro_f1']  = np.nanmean(f1_scores)


@ex.capture
def after_epoch_end(trainer, _run):
    _run.log_scalar(trainer.fold + "_train_loss",
                    trainer.cache['train_loss'].item())
    _run.log_scalar(trainer.fold + "_train_macro_f1",
                    trainer.cache['train_macro_f1'])
    _run.log_scalar(trainer.fold + "_val_loss",
                    trainer.cache['val_loss'].item())
    _run.log_scalar(trainer.fold + "_val_macro_f1",
                    trainer.cache['val_macro_f1'])


@ex.capture
def get_dir_name(model, optimizer, data, path, criterion, seed):
    name = model['model']
    name += '_' + optimizer['optimizer'] + '_' + str(optimizer['lr'])
    name += '_' + criterion['loss'] + '_' + criterion['weight']
    name += '_' + str(seed)
    print('Experiment code:', name)
    return name


if __name__ == '__main__':
    ex.run_commandline()
