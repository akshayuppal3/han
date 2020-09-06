from tensorflow.keras.layers import (
    Attention,
    Bidirectional,
    Concatenate,
    Dropout,
    Embedding,
    Dense,
    GRU,
    TimeDistributed,
    Input
)
from tensorflow.keras import Sequential, Model
from sklearn.model_selection import train_test_split

from train.attention import AttentionLayer

EMBED_SIZE = 100
MAX_SENTENCE_LEN = 580
MAX_WORD_LEN = 100


def get_attention_model(vocab_size, embedding_matrix, embed_size=100):
    word_input = Input(shape=(MAX_WORD_LEN,), dtype='int32', name='word_input')
    word_sequence = Embedding(
        vocab_size,
        embed_size,
        input_length=MAX_WORD_LEN,  # length of sentences in doc
        weights=[embedding_matrix],
        trainable=False)(word_input)

    ## attention at words
    word_gru = Bidirectional(GRU(
        50,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        return_sequences=True))(word_sequence)
    word_dense = Dense(100, activation='relu', name="word_dense")(word_gru)
    word_att, word_coeff = AttentionLayer(embed_size, True, name="word_attention")(word_dense)
    word_encoder = Model(inputs=word_input, outputs=word_att, name="word_encoder")

    ## attention at sentence level
    sent_input = Input(shape=(MAX_SENTENCE_LEN, MAX_WORD_LEN), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(word_encoder, name="sent_linking")(sent_input)
    sent_gru = Bidirectional(GRU(
        50,
        return_sequences=True))(sent_encoder)
    sent_dense = Dense(100, activation="relu", name="sent_dense")(sent_gru)
    sent_att, sent_coeff = AttentionLayer(embed_size, True, name="sent_attention")(sent_dense)
    sent_drop = Dropout(0.5)(sent_att)
    preds = Dense(1, activation="softmax", name="prediction")(sent_drop)

    # # model
    han_model = Model(sent_input, preds, name="han_model")
    han_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(word_encoder.summary())
    print(han_model.summary())

    return han_model
