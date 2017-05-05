from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, Callback
import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print(" First epoch has been initiated")
    def on_epoch_end(self, batch, logs={}):
        print(str(logs))
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        print(" Epoch: " + str(batch) + "- ACC: " + str(acc) + " LOSS: " + str(loss) + " VAL_ACC: " + str(val_acc))
