class DefaultConfigs(object):
    train_data = "/home/trinhnh1/Documents/kaggle/human-protein/input/train_all/" # where is your train data
    # train_data = "/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/external_data/img/train_all/"
    test_data = "/home/trinhnh1/Documents/kaggle/human-protein/input/test_jpg/"   # your test data
    sample_submission = "/home/trinhnh1/Documents/kaggle/human-protein/input/sample_submission.csv"
    train_kaggle_csv = "/home/trinhnh1/Documents/kaggle/human-protein/input/train.csv"
    train_external_csv = "/home/trinhnh1/Documents/kaggle/human-protein/input/external_data/img/train.csv"
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models"
    submit = "./submit/"
    model_name = "resnet50_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 1e-03
    lr_ft = 0.02
    batch_size = 24
    epochs = 45
    epochs_ft = 2
    seed = 2050
    kfold = 5

config = DefaultConfigs()
