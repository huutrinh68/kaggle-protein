class DefaultConfigs(object):
    root_path = "/home/trinhnh1/Documents/kaggle/human-protein/input"
    train_data = root_path + "/train_all_used_L_to_convert_to_gray/" # where is your train data
    test_data = root_path + "/test_jpg/"   # your test data
    sample_submission = root_path + "/sample_submission.csv"
    train_kaggle_csv = root_path + "/train.csv"
    train_external_csv = root_path + "/external_data/img/train.csv"
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "seresnet50_bcelog"
    seed = 2050
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.0001
    weight_decay = 0.000000001
    batch_size = 18
    epochs = 15
    thresold = 0.2
    n_tta = 5
    n_fold = 5

config = DefaultConfigs()
