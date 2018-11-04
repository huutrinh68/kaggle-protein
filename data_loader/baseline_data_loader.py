import keras
import numpy as np
from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist


class SimpleMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleMnistDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((-1, 28 * 28))
        self.X_test = self.X_test.reshape((-1, 28 * 28))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

class BaselineModelDataLoader(keras.utils.Sequence):
    def __init__(self, config, list_ids, labels, image_preprocessor):
        self.list_ids = list_ids
        self.labels = labels
        self.dim = (int(config.data.image_rows/config.data.row_scale_factor), 
                    int(config.data.image_cols/config.data.col_scale_factor))
        self.batch_size = config.trainer.batch_size
        self.n_channels = config.data.n_channels
        self.num_classes = config.data.num_classes
        self.shuffle = config.trainer.shuffle
        self.preprocessor = image_preprocessor
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        print(self.indexes)

    def get_label_per_image(self, identifier):
        return self.labels.loc[self.labels.Id == identifier].drop(
            ["Id", "Target", "number_of_targets"], axis=1).values

    def data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples' # X: (n_samples, *dim, n_channels)
        # Initilization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, identifier in enumerate(list_ids_temp):
            # Store sample
            image = self.preprocessor.load_image(identifier)
            image = self.preprocessor.preprocess(image)
            X[i] = image
            # Store class
            y[i] = self.get_label_per_image(identifier)
        return X, y

    def get_item(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of ids
        list_id_temp = [self.list_ids[k] for k in indexes]
        # Generate data
        X, y = self.data_generation(list_id_temp)
        return X, y



        
