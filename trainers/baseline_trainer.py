from base.base_trainer import BaseTrain
from utils.common import *


class BaseLineModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(BaseLineModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, "InceptionV3.h5"),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            EarlyStopping(monitor='val_loss', mode='min', patience=6)
        )

        self.callbacks.append(
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.0001)
        )
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp.name, workspace=self.config.exp.workspace)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self, warm_up=False):
        epochs = self.config.trainer.epochs
        if warm_up == True:
            epochs = self.config.trainer.warm_up_epochs

        history = self.model.fit_generator(
            self.data[0],
            steps_per_epoch=self.config.trainer.steps_per_epoch,
            validation_data=self.data[1],
            validation_steps=self.config.trainer.validation_steps,
            epochs=epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks)

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
