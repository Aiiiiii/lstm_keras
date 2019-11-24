import os
from models.finetuned_resnet import finetuned_resnet
from models.finetuned_vgg import finetuned_vgg
from utils.model_processing import model_processing

N_CLASSES = 15
IMSIZE = (200, 200, 3)


if __name__ == '__main__':
    # src_dir = '/home/lsz/AJ/vlair-lstm/ActionRecognition_rnn/data/UCF-Preprocessed'
    # dest_dir = '/home/lsz/AJ/vlair-lstm/ActionRecognition_rnn/data/CNN_Predicted'
    # weights_dir = '/home/lsz/AJ/vlair-lstm/ActionRecognition_rnn/models'
    # interval = '10s'
   


    dataname = 'milan'
    stepSize = '5s'
    padLen = 0.2  # 20%
    secondsPerFrame = 60  # second
    data_dir = '/home/lsz/AJ/vlair-lstm/data/%s/'%dataname
    src_dir =  os.path.join(data_dir, '%ds_%s'%(secondsPerFrame,stepSize))
    dest_dir = os.path.join(src_dir,'CNN_Predicted')    #'/home/lsz/AJ/vlair-lstm/data/milan15/%s/CNN_Predicted'%interval
    weights_dir = '/home/lsz/AJ/vlair-lstm/models'


    TIMESEQ_LEN = 5
    finetuned_resnet_weights = os.path.join(weights_dir, 'finetuned_vgg_%s_%ds_%s.h5'%(dataname,secondsPerFrame,stepSize)) # weight trained from train_CNN.py
    model = finetuned_vgg(include_top=False, weights_dir=finetuned_resnet_weights)
    model_processing(model, src_dir, dest_dir, TIMESEQ_LEN)