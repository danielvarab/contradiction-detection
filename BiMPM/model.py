from keras.layers import Input, LSTM, Dense, Embedding, merge, Bidirectional, Lambda, Flatten, Reshape, TimeDistributed
from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf

class MPL(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MPL, self).__init__(**kwargs)

    def build(self, input_shape):
        p_input_shape = input_shape[0]
        self.batch_size = p_input_shape[2]
        self.W = self.add_weight(shape=(self.output_dim, p_input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p = x[0] # shape (batch, sequence, d) .. (32, 10, 350)
        q = x[1] # shape (batch, sequence, d) .. (32, 10, 350)

        batch_distances = []
        for i in range(32):
            m = []
            for k in range(self.output_dim):
                # to implememnt the max match, https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L978
                weighted_p = p[i,:,:] * self.W[k]
                weighted_q = q[i,-1,:] * self.W[k] # full match
                bi_k_distances = cos_distance(weighted_q, weighted_p)
                m.append(bi_k_distances)
            m = K.concatenate(m)
            batch_distances.append(m)

        concatted = K.concatenate(batch_distances)
        return K.reshape(concatted, (32, 10, self.output_dim))

    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape # (l, d) where d is the sentence/sequence length
        return (shape1[0], shape2[1], self.output_dim)

def cosine_distance(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

class VanillaCosine(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(VanillaCosine, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: (batch, sentence_length, 100)

        word_distances = []
        for word_index in range(self.sentence_length):
            di = cosine_distance(p[:,word_index,:], q[:,-1,:]) # (32, 1)
            word_distances.append(di)
        result = K.stack(word_distances) # (10, 32, 1) => actually want (32,1,10), this is done below
        return tf.transpose(result,(1,2,0))


    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])

# sentence and words are capped at length 10 here, 15 is the word length
def build_model(sentence_length, word_length):
    word_a_input = Input(shape=(sentence_length, 300), name="word_sentence_A")
    word_b_input = Input(shape=(sentence_length, 300), name="word_sentence_B")
    char_a_input = Input(shape=(sentence_length, 15), name="char_sentence_A")
    char_b_input = Input(shape=(sentence_length, 15), name="char_sentence_B")

    # 27 is the vocab size, 20 is the embedding dimensionality, input length is straight forward the length of chars
    char_embedding = TimeDistributed(Embedding(27, 20, input_length=word_length))

    char_a_embs = char_embedding(char_a_input)
    char_b_embs = char_embedding(char_b_input)

    char_lstm = TimeDistributed(LSTM(50))

    char_sentence_a = char_lstm(char_a_embs)
    char_sentence_b = char_lstm(char_b_embs)

    sentence_a_emb = merge([word_a_input, char_sentence_a], mode='concat', name="sentence_a_emb")
    sentence_b_emb = merge([word_b_input, char_sentence_b], mode='concat', name="sentence_b_emb")

    fw_a_ctx, bw_a_ctx = Bidirectional(LSTM(100, return_sequences=True), merge_mode=None)(sentence_a_emb)
    fw_b_ctx, bw_b_ctx = Bidirectional(LSTM(100, return_sequences=True), merge_mode=None)(sentence_b_emb)

    forward_matchings = VanillaCosine()([fw_a_ctx, fw_b_ctx])
    backward_matchings = VanillaCosine()([bw_a_ctx, bw_b_ctx])

    forward_matching_aggregation = Bidirectional(LSTM(100), merge_mode="concat")(forward_matchings)
    backward_matching_aggregation = Bidirectional(LSTM(100), merge_mode="concat")(backward_matchings)

    matching_vector = merge([forward_matching_aggregation, backward_matching_aggregation], mode='concat')

    prediction_layer = Dense(1, activation='sigmoid')(matching_vector)

    model = Model(input=[word_a_input, char_a_input, word_b_input, char_b_input], output=prediction_layer)

    return model
