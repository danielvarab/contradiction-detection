from keras import backend as K
from keras.engine.topology import Layer

class Align(Layer):
    def __init__(self, normalize=False, trainable=False, **kwargs):
        self.normalize = normalize
        self.trainable = trainable
        super(Align, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # if self.trainable:
        #     self.weights = self.add_weight(shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        super(Align, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a, b = x

        # if we normalize, this layer outputs the cosine similarity
        if self.normalize:
            a = K.l2_normalize(a, axis=2)
            b = K.l2_normalize(b, axis=2)

        return K.batch_dot(a, b, axes=[2, 2])

    def compute_output_shape(self, input_shape):
        a_shape, b_shape = input_shape
        return (a_shape[0], a_shape[1], b_shape[1])

class Summarize(Layer):
    def __init__(self, trainable=False, **kwargs):
        self.trainable = trainable
        super(Summarize, self).__init__(**kwargs)

    def build(self, input_shape):
        # if self.trainable:
        #     self.kernel = self.add_weight(shape=(input_shape[1], input_shape[1]), initializer='uniform', trainable=True)
        super(Summarize, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x_shape = K.int_shape(x)
        bow = x[:,0,:]
        for i in range(1, x_shape[1]):
            # if self.trainable: word = self.kernel * x[:,i,:]
            # else : word = x[:,i,:]
            bow = bow + x[:,i,:]

        return bow

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
