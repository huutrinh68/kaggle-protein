class ImprovedModel(BaseLineModel):
    
    def __init__(self, modelparameter, my_metrics=[f1]):
        super().__init__(modelparameter)
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