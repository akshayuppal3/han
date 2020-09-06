import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention_dim=100, return_coefficients=False, **kwargs):
        super(AttentionLayer, self).__init__()
        self.support_masking = True
        self.return_coefficients = return_coefficients
        self.init = tf.random_normal_initializer()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = tf.Variable(self.init(shape=(input_shape[-1], self.attention_dim), dtype='float32'), trainable=True)
        self.u = tf.Variable(self.init(shape=(self.attention_dim, 1), dtype='float32'), trainable=True)
        self.b = tf.Variable(self.init(shape=(self.attention_dim,), dtype='float32'), trainable=True)

    def call(self, hit, mask=None):
        # calculation based from - Yang et. al.
        # Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

        uit = tf.matmul(hit, self.W) + self.b
        uit = tf.tanh(uit)

        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, -1)
        ait = tf.exp(ait)

        if mask is not None:
            ait *= tf.cast(mask, tf.floatx())

        ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
                       tf.keras.backend.floatx())
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = hit * ait

        if self.return_coefficients:
            return [tf.reduce_sum(weighted_input, axis=1), ait]  # ait are the coeffcients
        else:
            return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]
