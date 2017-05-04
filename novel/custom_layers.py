from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import merge
from keras.layers.normalization import BatchNormalization


class Align(Layer):
    def __init__(self, normalize=True, **kwargs):
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

class Aggregate(Layer):
    def __init__(self, operator=None, axis=None, **kwargs):
        self.operator = operator
        self.axis = axis
        super(Aggregate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Aggregate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.operator == "SUM":
            return K.sum(x, axis=self.axis)
        elif self.operator == "MAX":
            return K.max(x, axis=self.axis)
        elif self.operator == "MIN":
            return K.min(x, axis=self.axis)
        elif self.operator == "MEAN":
            return K.mean(x, axis=self.axis)
        else:
            raise AttributeError('operator is not valid {}'.format(self.operator))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

def _align(a, b, normalize):
    return Align(normalize)([a,b])

def _aggregate(x, op, axis):
    aggregation = Aggregate(op, axis)(x)
    return BatchNormalization()(aggregation)

#
# def sum_both_directions(x):
#     left = K.sum(x,axis=1)
#     right = K.sum(x,axis=2)
#     return K.concatenate([left, right][:], axis=-1)
#
# def max_both_directions(x):
#     left = K.max(x,axis=1)
#     right = K.max(x,axis=2)
#     return K.concatenate([left, right][:], axis=-1)
#
# def min_both_directions(x):
#     left = K.min(x,axis=1)
#     right = K.min(x,axis=2)
#     return K.concatenate([left, right][:], axis=-1)
#
# def mean_both_directions(x):
#     left = K.mean(x,axis=1)
#     right = K.mean(x,axis=2)
#     return K.concatenate([left, right][:], axis=-1)
#
#
# def both_directions_output_shape(input_shape):
#     return (input_shape[0], input_shape[1]*2)
