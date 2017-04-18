from keras.layers import Input, LSTM, Dense, Embedding, merge, Bidirectional, Lambda, Flatten, Reshape, TimeDistributed, Dropout
from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
from matching import *

# sentence and words are capped at length 10 here, 15 is the word length
def build_model(char_vocab_size, sentence_length, word_length):
    word_a_input = Input(shape=(sentence_length, 300), name="word_sentence_A")
    word_b_input = Input(shape=(sentence_length, 300), name="word_sentence_B")
    char_a_input = Input(shape=(sentence_length, word_length), name="char_sentence_A")
    char_b_input = Input(shape=(sentence_length, word_length), name="char_sentence_B")

    char_embedding = TimeDistributed(Embedding(char_vocab_size, 20, input_length=word_length))

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

    prediction_layer = Dense(3, activation='softmax')(matching_vector)

    model = Model(input=[word_a_input, char_a_input, word_b_input, char_b_input], output=prediction_layer)

    return model

def build_model_2(char_vocab_size, sentence_length, word_length):
    word_a_input = Input(shape=(sentence_length, 300), name="word_sentence_A")
    word_b_input = Input(shape=(sentence_length, 300), name="word_sentence_B")
    char_a_input = Input(shape=(sentence_length, word_length), name="char_sentence_A")
    char_b_input = Input(shape=(sentence_length, word_length), name="char_sentence_B")

    char_embedding = TimeDistributed(Embedding(char_vocab_size, 20, input_length=word_length))

    char_a_embs = char_embedding(char_a_input)
    char_b_embs = char_embedding(char_b_input)

    char_lstm = TimeDistributed(LSTM(50))

    char_sentence_a = char_lstm(char_a_embs)
    char_sentence_b = char_lstm(char_b_embs)

    sentence_a_emb = merge([word_a_input, char_sentence_a], mode='concat', name="sentence_a_emb")
    sentence_b_emb = merge([word_b_input, char_sentence_b], mode='concat', name="sentence_b_emb")

    # one direction
    fw_a_ctx = LSTM(100, return_sequences=True)(sentence_a_emb)
    fw_b_ctx = LSTM(100, return_sequences=True)(sentence_b_emb)

    fw_a_ctx = Dropout(.1)(fw_a_ctx)
    fw_b_ctx = Dropout(.1)(fw_b_ctx)

    pq_match_1 = FullMatch()([fw_a_ctx, fw_b_ctx])
    pq_match_2 = MaxPoolingMatch()([fw_a_ctx, fw_b_ctx])
    pq_match_3 = AttentiveMatch()([fw_a_ctx, fw_b_ctx])
    pq_match_4 = MaxAttentiveMatch()([fw_a_ctx, fw_b_ctx])

    qp_match_1 = FullMatch()([fw_b_ctx, fw_a_ctx])
    qp_match_2 = MaxPoolingMatch()([fw_b_ctx, fw_a_ctx])
    qp_match_3 = AttentiveMatch()([fw_b_ctx, fw_a_ctx])
    qp_match_4 = MaxAttentiveMatch()([fw_b_ctx, fw_a_ctx])

    pq_combined_matches = merge([pq_match_1, pq_match_2, pq_match_3, pq_match_4], mode="concat", concat_axis=1)
    qp_combined_matches = merge([qp_match_1, qp_match_2, qp_match_3, qp_match_4], mode="concat", concat_axis=1)

    p_to_q_fw_bw_aggr = Bidirectional(LSTM(100), merge_mode="concat")(pq_combined_matches)
    q_to_p_fw_bw_aggr = Bidirectional(LSTM(100), merge_mode="concat")(qp_combined_matches)

    p_to_q_fw_bw_aggr = Dropout(.1)(p_to_q_fw_bw_aggr)
    q_to_p_fw_bw_aggr = Dropout(.1)(q_to_p_fw_bw_aggr)

    matching_vector = merge([p_to_q_fw_bw_aggr, q_to_p_fw_bw_aggr], mode='concat')

    prediction_layer = Dense(3, activation='softmax', name="output")(matching_vector)

    model = Model(input=[word_a_input, char_a_input, word_b_input, char_b_input], output=prediction_layer)

    return model

def build_model_3
