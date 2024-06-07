import argparse
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from keras import backend as K
from keras.utils import custom_object_scope

def psnr_metric(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def convert_to_mosaic(image):

    if image.shape[2] != 3:
        raise ValueError("Image must be in RGB format")

    # Get every color channel
    I_red = image[:, :, 0]
    I_green = image[:, :, 1]
    I_blue = image[:, :, 2]
    # initialize mosaic image
    mosaic = np.zeros_like(image)

    # bayer pattern
    mosaic[::2, ::2, 1] = I_green[::2, ::2]   # green
    mosaic[1::2, 1::2, 1] = I_green[1::2, 1::2] # green
    mosaic[::2, 1::2, 2] = I_red[::2, 1::2]     # red
    mosaic[1::2, ::2, 0] = I_blue[1::2, ::2]    # blue

    # return the mosaic image which only have one channel
    return mosaic * (mosaic > 0)




def test(model, test_dir, sigma, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    name = []
    psnr = []
    ssim = []
    file_list = glob.glob('{}/*.tif'.format(test_dir))    ## if u use png, plz change the format to .png
    for file in file_list:
        img_clean = np.array(Image.open(file).convert('RGB'), dtype='float32') / 255.0
        img_test = convert_to_mosaic(img_clean)
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 3) 
        y_predict = model.predict(x_test)
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
        ssim_noise = compare_ssim(img_clean, img_test, data_range=1, multichannel=True, channel_axis=-1)
        ssim_denoised = compare_ssim(img_clean, img_out, data_range=1, multichannel=True, channel_axis=-1)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        filename = os.path.basename(file).split('.')[0] 
        name.append(filename)
        img_test_path = os.path.join(save_dir, '{}_sigma{}_psnr{:.2f}.tif'.format(filename, sigma, psnr_noise))
        img_out_path = os.path.join(save_dir, '{}_psnr{:.2f}.tif'.format(filename, psnr_denoised))
        img_test = Image.fromarray((img_test * 255).astype('uint8'))
        img_test.save(img_test_path)
        img_out = Image.fromarray((img_out * 255).astype('uint8'))
        img_out.save(img_out_path)
    
    psnr_avg = sum(psnr) / len(psnr)
    ssim_avg = sum(ssim) / len(ssim)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    pd.DataFrame({'name': np.array(name), 'psnr': np.array(psnr), 'ssim': np.array(ssim)}).to_csv(os.path.join(save_dir, 'metrics.csv'), index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory of test dataset')
    parser.add_argument('--sigma', type=int, default=25, help='Noise level')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')
    args = parser.parse_args()

    with custom_object_scope({'psnr_metric': psnr_metric}):
        model = load_model(args.model_path)
    test(model, args.test_dir, args.sigma, args.save_dir)