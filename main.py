from comet_ml import Experiment
from data_loader.baseline_data_loader import DataGenerator
from models.baseline_model import BaseLineModel
from trainers.baseline_trainer import BaseLineModelTrainer
from utils.common import *
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)
    
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



    #################################################################################################
    #******************************start create train-validation data ******************************#
    path_to_train = os.path.join(root_path, 'train')
    data = pd.read_csv(os.path.join(root_path, 'train.csv'))

    SIZE = config.data.img_cols
    n_channels = config.data.n_channels
    batch_size = config.trainer.batch_size

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path':os.path.join(path_to_train, name),
            'labels':np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    indexes = np.arange(train_dataset_info.shape[0])
    np.random.shuffle(indexes)

    train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=0)
    
    data_generator = DataGenerator(config)
    train_generator = data_generator.create_train(train_dataset_info[train_indexes], batch_size, (SIZE, SIZE, n_channels), augment=False)
    valid_generator = data_generator.create_train(train_dataset_info[valid_indexes], batch_size, (SIZE, SIZE, n_channels), augment=False)
    
    data_train_valid = []
    data_train_valid.append(train_generator)
    data_train_valid.append(valid_generator)
    #*******************************end create train-validation data *******************************#
    #################################################################################################



    #################################################################################################
    #*******************************   start create then train model *******************************#
    print('Create the model.')
    model = BaseLineModel(config).build_model()

    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(1e-03),
        metrics=["accuracy"])

    # train model
    print('Create the trainer')
    trainer = BaseLineModelTrainer(model, data_train_valid, config)
    trainer.train(warm_up=True)

    # train all layers
    for layer in model.layers:
        layer.trainable = True

    # compile model  
    model.compile(loss="binary_crossentropy",
                optimizer=Adam(lr=1e-4),
                metrics=["accuracy"])
    # train model
    trainer.train(warm_up=False)
    #******************************* end create then train model ***********************************#
    #################################################################################################



    #################################################################################################
    #***********************************start create submit *****************************************#
    # create submit
    submit = pd.read_csv(os.path.join(root_path, "sample_submission.csv"))
    predicted = []
    draw_predict = []
    model.load_weights(os.path.join(root_path, "working/InceptionV3.h5"))

    for name in tqdm(submit["Id"]):
        path = os.path.join(root_path, "test", name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))/255.
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)
        label_predict = np.arange(28)[score_predict>=0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit["Predicted"] = predicted
    np.save("draw_predict_InceptionV3.npy", score_predict)
    submit.to_csv('submit_InceptionV3.csv', index=False)
    #*************************************end create submit ****************************************#
    #################################################################################################

if __name__ == '__main__':
    main()