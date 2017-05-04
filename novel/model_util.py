from keras.layers.normalization import BatchNormalization
from custom_layers import *

def build_sentence_tensors(a, b, aggregation_op_we, aggregation_op_ae, align_op_we, align_op_ae):
    """
        a / sentence_a      :: (?, 42, 300)
        b / sentence_b      :: (?, 42, 300)
        aggregation_op_we   :: "SÙM", "MIN", "MAX", "None"
        aggregation_op_ae   :: "SÙM", "MIN", "MAX", "None"
        align_op_we         :: "SÙM", "MIN", "MAX", "None"
        align_op_ae         :: "SÙM", "MIN", "MAX", "None"
    """
    aggre_a_we = _aggregate(a, aggregation_op_we, axis=1)
    aggre_b_we = _aggregate(b, aggregation_op_we, axis=1)
    aggre_a_ae = _aggregate(a, aggregation_op_ae, axis=1)
    aggre_b_ae = _aggregate(b, aggregation_op_ae, axis=1)
    align_we = _align(a, b, align_op_we)
    align_ae = _align(a, b, align_op_we)

    sentence_representations = filter(lambda x : x != None, [ aggre_a_we, aggre_b_we, aggre_a_ae, aggre_b_ae, align_we, align_ae ])
    sentence_representations = list(sentence_representations)
    assert len(sentence_representations) > 0
    return sentence_representations


def _align(a, b, op):
    if op is None:
        return None
    return Align(op, axis)([a,b])

def _aggregate(x, op, axis):
    if op is None:
        return None
    aggregation = Aggregate(op, axis)(x)
    return BatchNormalization()(aggregation)
