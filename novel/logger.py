class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        print(str(logs))
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        logging.info("Epoch: " + str(batch) +"/" + str(EPOCHS)  + " ACC: " + str(acc) + " LOSS: " + str(loss) + " VAL_ACC: " + str(val_acc))
