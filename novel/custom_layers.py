from keras import backend as K
from keras.engine.topology import Layer

class Align(Layer):
    def __init__(self, normalize=False, trainable=False, **kwargs):
        self.normalize = normalize
        super(Align, self).__init__(**kwargs)

    def build(self, input_shape):
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

def max_both_directions(x):
    left = K.max(x,axis=1)
    right = K.max(x,axis=2)
    return K.concatenate([left, right], axis=1)

def max_both_directions_output_shape(input_shape):
    return (input_shape[0], input_shape[1]*2)
