import os

import tensorflow as tf
import numpy as np
import joblib 


file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = os.path.join(os.getcwd(), "shakespeare.txt")

tf.keras.utils.get_file(file_name, file_url)

with open(file_name, 'r') as file:
    text = file.read()

# creates letter id per distinct letter in text
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)

max_depth = len(tokenizer.word_index) # number of distinct letters
dataset_size = tokenizer.document_count # number of total letters
train_size = dataset_size * 9 // 10 

[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1

# since Char-RNN model is stateful a sequence in a batch
# should exactly start where the previous batch's sequence which has same index number left off
batch_size = 32
n_steps = 200
window_length = n_steps + 1
encoded_parts = np.array_split(encoded[:train_size], batch_size)

datasets = list()
for part in encoded_parts:
    dataset = tf.data.Dataset.from_tensor_slices(part)
    dataset = dataset.window(size=window_length, shift=n_steps, drop_remainder=True)

    # since window method of dataset returns a new dataset per window
    # the dataset object becomes a dataset of datasets
    # the inner datasets should be converted to regular tensors
    # following code line does exactly this
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    datasets.append(dataset)

dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:])) # creates labels
dataset = dataset.map(lambda X_batch, y_batch: (tf.one_hot(X_batch, depth=max_depth), y_batch))
dataset = dataset.prefetch(1)

model = tf.keras.Sequential()

# creation of stateful Char-RNN model
model.add(tf.keras.layers.GRU(256, return_sequences=True, stateful=True,
                              activation="tanh", dropout=0.2, recurrent_dropout=0.3, batch_input_shape=[batch_size, None, max_depth]))
model.add(tf.keras.layers.GRU(256, return_sequences=True, stateful=True, activation="tanh", dropout=0.2, recurrent_dropout=0.3))
model.add(tf.keras.layers.GRU(128, return_sequences=True, stateful=True, activation="tanh", dropout=0.2, recurrent_dropout=0.3))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_depth, activation="softmax")))

# the states should be reset
# at the beginning of each epoch
# due to the model is stateful
class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epochs, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])

model.save(os.path.join(os.getcwd(), "models", "stateful"))
model.save_weights(os.path.join(os.getcwd(), "weights", "shakespearean"))

joblib.dump(tokenizer, "tokenizer.save")