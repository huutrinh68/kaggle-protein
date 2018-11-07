# from data_loader.baseline_data_loader import BaselineModelDataLoader
# from models.baseline_model import BaseLineModel
# from models.improved_model import ImprovedModel
# from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
# from preprocessing.preprocessing import ImagePreprocessor
# from predicts.predict_generator import PredictGenerator
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.utils import get_args
# import numpy as np
# import os
# from sklearn.model_selection import RepeatedKFold
# import pandas as pd
# from utils.kernel import KernelSettings

# def main():
#     # capture the config path from the run arguments
#     # then process the json configuration file
#     try:
#         args = get_args()
#         config = process_config(args.config)
#     except:
#         print("missing or invalid arguments")
#         exit(0)

#     # create the experiments dirs
#     create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

#     # list out train & test files
#     train_path = os.path.join(config.data.root_path, "train")
#     test_path = os.path.join(config.data.root_path, "test")
#     train_files = os.listdir(train_path)
#     test_files = os.listdir(test_path)
#     percentage = np.round(len(test_files) / len(train_files) * 100)
#     print("The test set size turns out to be {} % compared to the train set.".format(percentage))

#     # K-fold cross-validation
#     splitter = RepeatedKFold(
#         n_splits=config.data.kfold_cv.n_splits,
#         n_repeats=config.data.kfold_cv.n_repeats,
#         random_state=config.data.kfold_cv.random_state
#     )

#     labels = \
#         pd.read_csv(os.path.join(config.data.root_path, "train.csv"))

#     partitions = []
#     for train_idx, test_idx in splitter.split(labels.index.values):
#         partition = {}
#         partition["train"] = labels.Id.values[train_idx]
#         partition["validation"] = labels.Id.values[test_idx]
#         partitions.append(partition)
#         print("TRAIN: {0} TEST: {1}".format(train_idx, test_idx))
#         print("TRAIN: {0} TEST: {1}".format(len(train_idx), len(test_idx)))

#     # preprocesing
#     preprocessor = ImagePreprocessor(config)

#     # training the baseline model on the first cv-fold
#     print('Create the data generator.')
#     partition = partitions[0]
#     training_generator = BaselineModelDataLoader(config, partition["train"], labels, preprocessor)
#     validation_generator = BaselineModelDataLoader(config, partition["validation"], labels, preprocessor)
#     predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)

#     # run computation and store results as csv
#     kernelsettings = KernelSettings(fit_baseline=False, fit_improved_baseline=True)
#     if kernelsettings.fit_improved_baseline == True:
#         print("test")
#         model = ImprovedModel(config)
#         # model.build_model()
#         # model.compile_model()
#         # model.set_generators(training_generator, validation_generator)
#         # history = model.learn()
#         # proba_predictions = model.predict(validation_generator)
#         # #model.save("improved_model.h5")
#         # improved_proba_predictions = pd.DataFrame(proba_predictions, columns=wishlist)
#         # improved_proba_predictions.to_csv("improved_predictions.csv")
#     # if you already have done a baseline fit once, 
#     # you can load predictions as csv and further fitting is not neccessary:
#     # else:
#     #     baseline_proba_predictions = pd.read_csv("../input/protein-atlas-eab-predictions/baseline_predictions.csv", index_col=0)
        
#     # print('Create the model.')
#     # model = SimpleMnistModel(config)

#     # print('Create the trainer')
#     # trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)

#     # print('Start training the model.')
#     # trainer.train()


# if __name__ == '__main__':
#     main()


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from utils.common import *

print(os.listdir(root_path))
# Any results you write to the current directory are saved as output.

path_to_train = os.path.join(root_path, 'train')
data = pd.read_csv(os.path.join(root_path, 'train.csv'))

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

class DataGenerator():
    
    def create_train(self, dataset_info, batch_size, shape, augment=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                X_train_batch = []
                tmp = dataset_info[start:end]
                y_train_batch = np.zeros((len(tmp), 28))
                for i in range(len(tmp)):
                    image = self.load_image(tmp[i]["path"], shape)
                    if augment == True:
                        image = self.augment(image)
                    X_train_batch.append(image/255.)
                    y_train_batch[i][tmp[i]["labels"]] = 1
                yield np.array(X_train_batch), y_train_batch

    def load_image(self, path, shape):
        image_red_ch = Image.open(path+"_red.png")
        image_yellow_ch = Image.open(path+"_yellow.png")
        image_green_ch = Image.open(path+"_green.png")
        image_blue_ch = Image.open(path+"_blue.png")
        # image = np.stack((np.array(image_red_ch), np.array(image_yellow_ch), np.array(image_green_ch), np.array(image_blue_ch)), -1)
        image = np.stack((np.array(image_red_ch), np.array(image_yellow_ch), np.array(image_green_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image
    
    def augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])], random_order=True)
        image_aug = augment_img.augment_image(img)
        return image_aug
    



def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x =  Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model



epochs = 30; batch_size = 16

checkpoint = ModelCheckpoint(os.path.join(root_path, 'working/InceptionV3.h5'), monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min', save_weights_only=True)
early = EarlyStopping(monitor='val_loss',
                     mode='min',
                     patience=6)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                  verbose=1, mode='auto', min_delta=0.0001)
callbacks_list = [checkpoint, early, reduceLROnPlat]

indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)

train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=0)
SIZE = 299
data_generator = DataGenerator()
train_generator = data_generator.create_train(train_dataset_info[train_indexes], batch_size, (SIZE, SIZE, 3), augment=False)
valid_generator = data_generator.create_train(train_dataset_info[valid_indexes], batch_size, (SIZE, SIZE, 3), augment=False)

# warm up model
model = create_model(input_shape=(SIZE, SIZE, 3), n_out=28)

for layer in model.layers:
    layer.trainable = False
    
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

model.compile(loss="binary_crossentropy",
              optimizer=Adam(1e-03),
              metrics=["accuracy"]
)

# model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
                    validation_data=valid_generator,
                    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
                    epochs=2,
                    verbose=1)

# train all layers
for layer in model.layers:
    layer.trainable = True
    
model.compile(loss="binary_crossentropy",
              optimizer=Adam(lr=1e-4),
              metrics=["accuracy"])

model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
                    validation_data=valid_generator,
                    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list)

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