from keras.layers import Input, LSTM, Dense, Embedding, merge, Bidirectional, Lambda, Flatten, Reshape, TimeDistributed
from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

def cosine_distance(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

class FullMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(FullMatch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: (batch, sentence_length, 100)

        word_distances = [cosine_distance(p[:,word_index,:], q[:,-1,:]) for word_index in range(self.sentence_length)]
        result = K.stack(word_distances)
        return tf.transpose(result,(1,2,0))


    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape2[1])

class MaxPoolingMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(MaxPoolingMatch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: (batch, sentence_length, 100)

        word_distances = []
        for i in range(self.sentence_length): # i :: word in p
            dij = [cosine_distance(p[:,i,:], q[:,j,:]) for j in range(self.sentence_length)] # list[(32,1)] of length 10
            maxed = K.concatenate(dij, axis=1) # (32, 10)
            maxed = K.max(maxed, axis=1, keepdims=True) # (32, 1)
            word_distances.append(maxed)
        result = K.stack(word_distances) # (10, 32, 1) => actually want (32,1,10), this is done below
        return tf.transpose(result,(1,2,0))


    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])

class AttentiveMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(AttentiveMatch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: 2x(batch, sentence_length, 100)

        mean_vecs = []
        for i in range(self.sentence_length): # i :: word in p
            di = [cosine_distance(p[:,i,:], q[:,j,:]) for j in range(self.sentence_length)] # list[(32,1)]
            di = K.expand_dims(K.concatenate(di, axis=1), 2) # (32, 10) for ith word => (expand) => (32, 10, 1)
            attentive_vector = q[:,:,:] * di # (32, 10, 100)*(32, 10, 1) => (32, 10, 100)
            mean_vec = K.mean(attentive_vector, axis=1, keepdims=True) # => (32, 1, 100)
            mean_vecs.append(mean_vec)
        mean_vecs = K.concatenate(mean_vecs, axis=1) # => (32, 10, 100)

        word_distances = [cosine_distance(p[:,i,:], mean_vecs[:,i,:]) for i in range(self.sentence_length)]
        result = K.stack(word_distances)
        return tf.transpose(result,(1,2,0))

    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])

class MaxAttentiveMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(MaxAttentiveMatch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: 2x(batch, sentence_length, 100)

        mean_vecs = []
        for i in range(self.sentence_length): # i :: word in p
            di = [cosine_distance(p[:,i,:], q[:,j,:]) for j in range(self.sentence_length)] # list[(32,1)]
            di = K.expand_dims(K.concatenate(di, axis=1), 2) # (32, 10) for ith word => (expand) => (32, 10, 1)
            attentive_vector = q[:,:,:] * di # (32, 10, 100)*(32, 10, 1) => (32, 10, 100)
            mean_vec = K.max(attentive_vector, axis=1, keepdims=True) # => (32, 1, 100)
            mean_vecs.append(mean_vec)
        mean_vecs = K.concatenate(mean_vecs, axis=1) # => (32, 10, 100)

        word_distances = [cosine_distance(p[:,i,:], mean_vecs[:,i,:]) for i in range(self.sentence_length)]
        result = K.stack(word_distances)
        return tf.transpose(result,(1,2,0))

    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])

if __name__ == "__main__":
    # dummy data
    data_sentences_a = np.random.uniform(-1, 1, (3200,10,100))
    data_sentences_b = np.random.uniform(-1, 1, (3200,10,100))
    labels = np.random.randint(3, size=(3200,3))

    print(data_sentences_a.shape, data_sentences_b.shape, labels.shape)

    sentence_a = Input(shape=(10, 100), name="a")
    sentence_b = Input(shape=(10, 100), name="b")

    match_1 = FullMatch()([sentence_a, sentence_b])
    match_2 = MaxPoolingMatch()([sentence_a, sentence_b])
    match_3 = AttentiveMatch()([sentence_a, sentence_b])
    match_4 = MaxAttentiveMatch()([sentence_a, sentence_b])

    combined_matches = merge([match_1, match_2, match_3, match_4], mode="concat", concat_axis=1)

    matching_aggregation = LSTM(100)(combined_matches)
    prediction_layer = Dense(3, activation="softmax")(matching_aggregation)
    model = Model(input=[sentence_a, sentence_b], output=prediction_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit([data_sentences_a, data_sentences_b], labels, nb_epoch=10, batch_size=32)
