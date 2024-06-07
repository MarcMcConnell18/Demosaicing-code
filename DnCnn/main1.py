import argparse
import logging
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
import cv2
from keras import backend as K
#import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import load_model
from keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import models
#from util import *


## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--val_data_dir', default='./data/npy_data_val/', type=str, help='directory of validation data')
parser.add_argument('--train_data', default='./data/npy_data/clean_patches.npy', type=str, help='path of train data')
parser.add_argument('--test_dir', default='./data/Test/Set24', type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=25, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
args = parser.parse_args()

if not args.only_test:
    save_dir = './snapshot/save_'+ args.model + '_' + 'sigma' + str(args.sigma) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # log
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename=save_dir+'info.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(args)
    
else:
    save_dir = '/'.join(args.pretrain.split('/')[:-1]) + '/'

def load_train_data(npy_dir):
    while True:
        file_list = sorted(glob.glob(npy_dir + '*.npy'))
        for file in file_list:
            data = np.load(file)
            yield data

def load_training_data():

    data = np.load(args.train_data)
    logging.info('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))
    
    return data

def step_decay(epoch):
    
    initial_lr = args.lr
    if epoch<50:
        lr = initial_lr
    else:
        lr = initial_lr/10
    
    return lr

def convert_to_mosaic(image):

    if image.shape[2] != 3:
        raise ValueError("Image must be in RGB format")

    # Get every color channel
    I_red = image[:, :, 0]
    I_green = image[:, :, 1]
    I_blue = image[:, :, 2]
    # initialize mosaic image
    mosaic = np.zeros_like(image)

# ### red=2 blue=0
#     pattern = [
#         [1, 0, 1, 1, 2, 1],
#         [2, 1, 2, 0, 1, 0],
#         [1, 0, 1, 1, 2, 1],
#         [1, 2, 1, 1, 0, 1],
#         [0, 1, 0, 2, 1, 2],
#         [1, 2, 1, 1, 0, 1]
#     ]

    # # Apply X-Trans pattern
    # for i in range(6):
    #     for j in range(6):
    #         if pattern[i][j] == 0:
    #             mosaic[i::6, j::6, 0] = I_red[i::6, j::6]
    #         elif pattern[i][j] == 1:
    #             mosaic[i::6, j::6, 1] = I_green[i::6, j::6]
    #         elif pattern[i][j] == 2:
    #             mosaic[i::6, j::6, 2] = I_blue[i::6, j::6]
    # #RGBW #3
    # mosaic[0::4, 2::4, 2] = I_red[0::4, 2::4]   # 3,4
    # mosaic[2::4, 2::4, 2] = I_red[2::4, 2::4]   # 4,3

    # # green
    # mosaic[0::4, 0::4, 1] = I_green[0::4, 0::4]   # 1,4
    # mosaic[1::4, 2::4, 1] = I_green[1::4, 2::4] # 2,3
    # mosaic[2::4, 0::4, 1] = I_green[2::4, 0::4] # 3,2
    # mosaic[3::4, 2::4, 1] = I_green[3::4, 2::4]   # 4,1

    # # blue
    # mosaic[1::4, 0::4, 0] = I_blue[1::4, 0::4]    # 1,2
    # mosaic[3::4, 0::4, 0] = I_blue[3::4, 0::4]    # 2,1

    # # white
    # mosaic[:, 1::2] = image[:, 1::2].mean(axis=2, keepdims=True)
    
    # #RGBW #2
    # mosaic[0::4, 2::4, 2] = I_red[0::4, 2::4]   
    # mosaic[1::4, 2::4, 2] = I_red[1::4, 2::4]   

    # # Green
    # mosaic[0::4, 0::4, 1] = I_green[0::4, 0::4]   
    # mosaic[1::4, 0::4, 1] = I_green[1::4, 0::4] 
    # mosaic[2::4, 2::4, 1] = I_green[2::4, 2::4] 
    # mosaic[3::4, 2::4, 1] = I_green[3::4, 2::4] 

    # # Blue
    # mosaic[2::4, 0::4, 0] = I_blue[2::4, 0::4]    
    # mosaic[3::4, 0::4, 0] = I_blue[3::4, 0::4]   

    # # White
    # mosaic[:, 1::2] = image[:, 1::2].mean(axis=2, keepdims=True)
    
    # #RGBW #1
    # mosaic[2::4, 3::4, 2] = I_red[2::4, 3::4]  
    # mosaic[3::4, 2::4, 2] = I_red[3::4, 2::4]   

    # #Green
    # mosaic[::4, 3::4, 1] = I_green[::4, 3::4]   
    # mosaic[1::4, 2::4, 1] = I_green[1::4, 2::4] 
    # mosaic[2::4, 1::4, 1] = I_green[2::4, 1::4] 
    # mosaic[3::4, ::4, 1] = I_green[3::4, ::4]  

    # # blue
    # mosaic[::4, 1::4, 0] = I_blue[::4, 1::4]   
    # mosaic[1::4, ::4, 0] = I_blue[1::4, ::4]   
    # # white
    # mosaic[0::2, 0::2] = image[0::2, 0::2].mean(axis=2, keepdims=True)
    # mosaic[1::2, 1::2] = image[1::2, 1::2].mean(axis=2, keepdims=True)

    # # RGBW bayer pattern
    # mosaic[::2, ::2] = image[::2, ::2].mean(axis=2, keepdims=True)  # white
    # mosaic[1::2, 1::2, 1] = I_green[1::2, 1::2] # green
    # mosaic[::2, 1::2, 2] = I_red[::2, 1::2]     # red
    # mosaic[1::2, ::2, 0] = I_blue[1::2, ::2]    # blue
    # bayer pattern
    mosaic[::2, ::2, 1] = I_green[::2, ::2]   # green
    mosaic[1::2, 1::2, 1] = I_green[1::2, 1::2] # green
    mosaic[::2, 1::2, 2] = I_red[::2, 1::2]     # red
    mosaic[1::2, ::2, 0] = I_blue[1::2, ::2]    # blue

    # return the mosaic image which only have one channel
    return mosaic * (mosaic > 0)


def train_datagen(y_, batch_size=8):
    # y_ is the tensor of clean patches
    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y_[indices[i:i+batch_size]]
            ge_batch_x = np.array([convert_to_mosaic(img) for img in ge_batch_y])
            yield ge_batch_x, ge_batch_y
  

def val_datagen(batch_size=8):
    val_original_gen = load_train_data(args.val_data_dir)
    while True:
        for val_original_data in val_original_gen:
            val_original_data = val_original_data.reshape((val_original_data.shape[0], val_original_data.shape[1], val_original_data.shape[2], 3))
            val_original_data = val_original_data.astype('float32') / 255.0

            for i in range(0, len(val_original_data), batch_size):
                ge_batch_y = val_original_data[i:i + batch_size]
                # noise =  np.random.normal(0, args.sigma/255.0, ge_batch_y.shape)    # noise
                ge_batch_x = np.array([convert_to_mosaic(img) for img in ge_batch_y])     
                noise =  np.random.normal(0, args.sigma/255, ge_batch_x.shape)    # noise


                ge_batch_x = ge_batch_x + noise            
                yield ge_batch_x, ge_batch_y

def psnr_metric(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def train():
    data = load_training_data()
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2], 3))
    data = data.astype('float32')/255.0 
    batch_size = args.batch_size 

    val_total_patches_per_file = 5000  
    val_number_of_files = 17  
    val_total_patches = val_total_patches_per_file * val_number_of_files
    val_steps_per_epoch = val_total_patches / batch_size

    # model selection
    if args.pretrain:   model = load_model(args.pretrain, compile=False)
    else:   
        if args.model == 'DnCNN': model = models.DnCNN()
    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'], metrics = [psnr_metric])
    
    # use call back functions
    ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_psnr', 
                    verbose=0, save_freq='epoch')
    csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # create self-defined call back
    test_callback = TestAfterEachEpoch()
    # train 
    history = model.fit(train_datagen(data, batch_size=args.batch_size),
                    steps_per_epoch=len(data)//args.batch_size,
                    epochs=args.epoch,
                    verbose=1,
                    validation_data=val_datagen(batch_size=args.batch_size),
                    validation_steps=val_steps_per_epoch, 
                    callbacks=[ckpt, csv_logger, lr, test_callback])    
    return model



class TestAfterEachEpoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('Testing after epoch: {}'.format(epoch + 1))
        test(self.model)

def test(model):
    
    print('Start to test on {}'.format(args.test_dir))
    out_dir = save_dir + args.test_dir.split('/')[-1] + '/'
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
    name = []
    psnr = []
    ssim = []
    file_list = glob.glob('{}/*.png'.format(args.test_dir))
    for file in file_list:
        # read image
        img_clean = np.array(Image.open(file).convert('RGB'), dtype='float32') / 255.0
        img_test = convert_to_mosaic(img_clean)
        img_test = img_test.astype('float32')
        # predict
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 3) 
        y_predict = model.predict(x_test)
        # calculate numeric metrics
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
        ssim_noise = compare_ssim(img_clean, img_test, data_range=1, multichannel=True, channel_axis = -1)
        ssim_denoised = compare_ssim(img_clean, img_out, data_range=1, multichannel=True, channel_axis = -1)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        # save images
        filename = os.path.basename(file).split('.')[0] 
        name.append(filename)
        img_test_path = os.path.join(out_dir, '{}_sigma{}_psnr{:.2f}.png'.format(filename, args.sigma, psnr_noise))
        img_out_path = os.path.join(out_dir, '{}_psnr{:.2f}.png'.format(filename, psnr_denoised))
        img_test = Image.fromarray((img_test * 255).astype('uint8'))
        img_test.save(img_test_path)
        img_out = Image.fromarray((img_out * 255).astype('uint8'))
        img_out.save(img_out_path)
    
    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    print(len(name), len(psnr), len(ssim))
    pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 'ssim':np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)
    
if __name__ == '__main__':   
    
    if args.only_test:
        model = load_model(args.pretrain, compile=False)
        test(model)
    else:
        model = train()
        #test(model)       
