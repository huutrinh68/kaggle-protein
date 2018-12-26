class DefaultConfigs(object):
    root_path = "/home/trinhnh1/Documents/kaggle/human-protein/input"
    train_data = root_path + "/train_all/" # where is your train data
    test_data = root_path + "/test_jpg/"   # your test data
    sample_submission = root_path + "/sample_submission.csv"
    train_kaggle_csv = root_path + "/train.csv"
    train_external_csv = root_path + "/external_data/img/train.csv"
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "resnet50_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.0001
    batch_size = 32
    epochs = 100

config = DefaultConfigs()
