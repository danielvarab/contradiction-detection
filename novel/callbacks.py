from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, Lambda, concatenate, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers.merge import Concatenate, Dot, maximum
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        print(str(logs))
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        logging.info("Epoch: " + str(batch) +"/" + str(EPOCHS)  + " ACC: " + str(acc) + " LOSS: " + str(loss) + " VAL_ACC: " + str(val_acc))
