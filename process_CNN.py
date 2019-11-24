import os
from models.finetuned_vgg import finetuned_vgg
from utils.model_processing import model_processing

N_CLASSES = 15
IMSIZE = (200, 200, 3)


if __name__ == '__main__':
    data_dir = './data/'
    src_dir =  os.path.join(data_dir)
    dest_dir = os.path.join(src_dir,'CNN_Predicted')    #'/home/lsz/AJ/vlair-lstm/data/milan15/%s/CNN_Predicted'%interval
    weights_dir = './models'

    TIMESEQ_LEN = 5
    finetuned_resnet_weights = os.path.join(weights_dir, 'finetuned_vgg.h5') # weight trained from train_CNN.py
    model = finetuned_vgg(include_top=False, weights_dir=finetuned_resnet_weights)
    model_processing(model, src_dir, dest_dir, TIMESEQ_LEN)