import os
import numpy as np
from utils.utils import sequence_generator, get_data_list, get_class_index, get_train_test, get_seqLen
import keras.callbacks
from models import RNN
from keras.optimizers import SGD, Adam

N_CLASSES = 15
BatchSize = 32
SEQ_LEN = 5

def fit_model(model, train_data, test_data, weights_dir, input_shape):
    try:
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print('Load weights')
        train_generator = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES) 
        test_generator = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=2, mode='min')
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/try', histogram_freq=0, write_graph=True, write_images=True)
        adam = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit_generator(
            train_generator,
            steps_per_epoch=int(np.floor(len(train_data)/BatchSize)),
            epochs=2000,
            validation_data=test_generator,
            validation_steps=int(np.floor(len(test_data)/BatchSize)),
            verbose=2,
            callbacks=[checkpointer, earlystopping, tensorboard]
        )
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = './data'
    processed_dir = os.path.join(data_dir)
    frame_dir = os.path.join(data_dir, processed_dir, 'videos_npy')
    weights_dir = './models'
    SEQ_LEN = get_seqLen(os.path.join(processed_dir,'data_videoName.npy'))
    class_index = get_class_index(data_dir)
    # split train and test data
    data, label = get_data_list(processed_dir, frame_dir, SEQ_LEN, class_index)
    train_data, test_data = get_train_test(data, label, processed_dir, class_index, cnn_predict=False)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    CNN_output = 1024
    input_shape = (SEQ_LEN, CNN_output)
    rnn_weights_dir = os.path.join(weights_dir, 'rnn.h5')
    RNN_model = RNN.RNN(rnn_weights_dir, CNN_output)
    fit_model(RNN_model, train_data, test_data, rnn_weights_dir, input_shape)
