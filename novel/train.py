import numpy

from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate, Dot
from keras.models import Model

from custom_layers import *

SENTENCE_MAX_LEN = 42
WORD_DIM = 300

s1 = Input(shape=(SENTENCE_MAX_LEN, WORD_DIM), name="sentence_a")
s2 = Input(shape=(SENTENCE_MAX_LEN, WORD_DIM), name="sentence_b")

alignment = Align()([s1,s2])

aggregation = Summarize(trainable=True)(alignment)

predictions = Dense(10, activation='softmax')(aggregation)

model = Model(inputs=[s1,s2], outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# model.fit(data, labels)  # starts training
