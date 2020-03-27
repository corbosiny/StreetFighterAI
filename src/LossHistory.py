from tensorflow.python import keras

class LossHistory(keras.callbacks.Callback):
    """A class for keras to use to store training losses for the model to use:
       1. initialize a LossHistory object inside your agent
       2. and put callbacks= [self.lossHistory] in the model.fit() call
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
