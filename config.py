class DefaultConfigs(object):
    train_data = "/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/train_all/" # where is your train data
    # train_data = "/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/external_data/img/train_all/"
    test_data = "/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/test_jpg/"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 1e-04
    lr_ft = 0.02
    batch_size = 32
    epochs = 100
    epochs_ft = 2
    seed = 2050

config = DefaultConfigs()
