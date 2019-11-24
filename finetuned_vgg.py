import os
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras import applications

N_CLASSES = 15  
IMSIZE = (200, 200, 3)


def finetuned_vgg(include_top, weights_dir):
    '''

    :param include_top: True for training, False for generating intermediate results for
                        LSTM cell
    :param weights_dir: path to load finetune_vgg.h5
    :return:
    '''
    base_model  = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMSIZE)
    x = base_model.output
    x = Flatten()(x)
    # x = Dense(2048, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model


if __name__ == '__main__':
    model = finetuned_vgg(include_top=True, weights_dir='')
    print(model.summary())