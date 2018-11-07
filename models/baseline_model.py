from base.base_model import BaseModel
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam

class BaseLineModel(BaseModel):
    
    def __init__(self, config):
        super(BaseLineModel, self).__init__(config)
        img_rows = config.data.img_rows
        img_cols = config.data.img_cols
        n_channels = config.data.n_channels
        self.input_shape = (img_rows, img_cols, n_channels)
        self.num_classes = config.data.num_classes
    
    def build_model(self):
        input_tensor = Input(shape=self.input_shape)
        base_model = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=self.input_shape)
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
        x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='sigmoid')(x)
        model = Model(input_tensor, output)

        # warm up model
        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.layers[-4].trainable = True
        model.layers[-5].trainable = True
        model.layers[-6].trainable = True

        return model