import os
import numpy as np
import keras.callbacks
from .utils import image_from_sequence_generator, sequence_generator, get_data_list, get_train_test, get_class_index, get_seqLen
from models.finetuned_vgg import finetuned_vgg
from keras.optimizers import SGD
import matplotlib.pyplot as plt


N_CLASSES = 15
# SEQ_LEN = 5
BatchSize = 32


def fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=False):
    try:
        train_generator = image_from_sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        test_generator = image_from_sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        print('Start fitting model')
        # while True:
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=2, mode='auto')
        reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, verbose=1, epsilon=1e-4,mode='min')
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/try', histogram_freq=0, write_graph=True, write_images=True)
        history = model.fit_generator(
            train_generator,
            # steps_per_epoch=200,
            steps_per_epoch=int(np.floor(len(train_data)/BatchSize)),
            epochs=200, 
            validation_data=test_generator,
            validation_steps=int(np.floor(len(test_data)/BatchSize)),   #200
            verbose=2,
            callbacks=[checkpointer, tensorboard, earlystopping,reduce_lr_loss]
        )
        # Plot the accuracy and loss curves
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))
        plt.figure()
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        print('Training is interrupted')



if __name__ == '__main__':
    data_dir = './data/'
    processed_dir = os.path.join(data_dir)
    frame_dir = os.path.join(data_dir, processed_dir, 'videos_npy')
    weights_dir = './models'
    SEQ_LEN = get_seqLen(os.path.join(processed_dir,'data_videoName.npy'))

    class_index = get_class_index(data_dir)
    # load RGB mean of training dataset
    # mean = np.load(os.path.join(processed_dir,'mean.npy'))
    # split train and test data
    data, label = get_data_list(processed_dir, frame_dir, SEQ_LEN, class_index, mean)
    train_data, test_data = get_train_test(data, label, processed_dir, class_index, cnn_predict=True)
    input_shape = (SEQ_LEN, 200, 200, 3)
    weights_dir = os.path.join(weights_dir, 'finetuned_vgg')
    model = finetuned_vgg(include_top=True, weights_dir=weights_dir)
    fit_model(model, train_data, test_data, weights_dir, input_shape)