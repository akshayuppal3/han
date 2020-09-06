import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import clean_text, tokenize_sent, sent_tokenize_pad
from train.train import get_attention_model
from util import load_embedding_matrix

input_path = "/Users/akshay.uppalimanage.com/Desktop/data/data_artifactory/doc/doc_level_data_v_0.2.csv"
glove_100d_path = "/Users/akshay.uppalimanage.com/Desktop/data/embeddings_pretrained/glove/glove.6B.100d.txt"

EMBED_SIZE = 100
MAX_WORD_LEN = 100
MAX_SENTENCE_LEN = 580

label = "Lease Agreement"


def main():
    df = pd.read_csv(input_path)
    df_sample = df.sample(n=1000, random_state=46)
    df_sample.text = df_sample.text.progress_apply(clean_text)

    vocab_text = df_sample.text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocab_text)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size = ", vocab_size)

    embedding_matrix = load_embedding_matrix(glove_100d_path, tokenizer, vocab_size, EMBED_SIZE)

    # splitting docs into sentences
    df_upd = df_sample.copy()
    df_upd.text = df_upd.text.progress_apply(tokenize_sent)

    padded_doc = df_upd.text.progress_apply(sent_tokenize_pad)

    X = pad_sequences(padded_doc, maxlen=MAX_SENTENCE_LEN)
    y = list(df_upd.apply(lambda x: 1 if x["type"] == label else 0, axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

    print("training the model")
    han_model = get_attention_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        embed_size=EMBED_SIZE
    )
    hist = han_model.fit(X_train, y_train, epochs=7, batch_size=32, validation_split=0.2)


if __name__ == '__main__':
    main()
