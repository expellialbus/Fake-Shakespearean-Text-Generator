import os

import tensorflow as tf
import numpy as np
import joblib

def preprocess(text):
    X = np.array(tokenizer.texts_to_sequences([text])) - 1
    return tf.one_hot(X, max_depth)

def next_char(text, temperature=1):
    X_test = preprocess(text)
    y_test = model.predict(X_test)[0, -1:, :]
    rescaled_logits = tf.math.log(y_test) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=100, temperature=1):
    """
    Completes text with predicted characters by model

    Parameters
    ----------
    text : str
           characters to predict next character

    n_chars : int
              number of characters will be generated

    temperature : float
                  parameter to control diversity of random chosen characters
                  0 - gets characters with high probability
                  1 - will give characters equal probability
    """
    
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

tokenizer = joblib.load("tokenizer.save")
max_depth = len(tokenizer.word_index)

# stateles model (for more information check out the README.md file)
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(256, return_sequences=True, input_shape=[None, max_depth]),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_depth, activation="softmax"))
])

model.build(tf.TensorShape([None, None, max_depth]))
model.load_weights(os.path.join(os.getcwd(), "weights", "shakespearean"))

print(complete_text("b")) # example usage

model.save(os.path.join(os.getcwd(), "models", "stateless"))