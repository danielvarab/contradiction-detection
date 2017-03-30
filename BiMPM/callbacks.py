from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, test_input, test_output):
        self.test_input = test_input
        self.test_output = test_output

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.test_input, self.test_output, verbose=0)
        print('\n\t Testing loss: {}, acc: {}\n'.format(loss, acc))
