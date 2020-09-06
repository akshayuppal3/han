from numpy import asarray
from numpy import zeros


def load_embedding_matrix(embedding_path, tokenizer, vocab_size, dim):
    embedding_index = dict()
    f = open(embedding_path)
    for line in f:
        values = line.split()
        word = values[0]
        embedding_index[word] = asarray(values[1:], dtype=float)

    embedding_matrix = zeros((vocab_size, dim))
    for word, index in tokenizer.word_index.items():
        if word in embedding_index:
            embedding_matrix[index] = embedding_index[word]

    return embedding_matrix
