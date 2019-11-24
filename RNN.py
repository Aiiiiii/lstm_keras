
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM

N_CLASSES = 15
SequenceLength = 5


def RNN(weights_dir, CNN_output):
    model = Sequential()

    model.add(LSTM(256, return_sequences=False, input_shape=(SequenceLength, CNN_output)))
    model.add(Dropout(0.9))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.summary()

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)

    return model
