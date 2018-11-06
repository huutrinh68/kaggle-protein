from models.baseline_model import BaseLineModel

import keras.backend as K

def base_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def f1_min(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.min(f1)

def f1_max(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.max(f1)

def f1_mean(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.mean(f1)

def f1_std(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.std(f1)

class ImprovedModel(BaseLineModel):
    
    def __init__(self, config, my_metrics=[f1_mean, f1_std, f1_min, f1_max]):
        super(ImprovedModel, self).__init__(config)
        self.my_metrics = my_metrics
    
    def learn(self):
        self.history = TrackHistory()
        return self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.params.n_epochs, 
            use_multiprocessing=True,
            workers=8,
            callbacks = [self.history]
        )