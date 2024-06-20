import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pylab as plt
from PIL import Image
import os
import time
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import sys
sys.path.insert(0, 'code')
from Utils_inverse_prob import *
from network import *
from algorithm_inv_prob import *
# %matplotlib inline


# Print a list of pre-trained denoiser architectures
print( os.listdir('denoisers'))

# Choose an architecture from the list
architecture = 'BF_CNN'


# Now, choose the range of noise used during training from the list below

training_noise='noise_range_0to100' 

denoiser = load_denoiser(architecture=architecture, 
                         grayscale='False', 
                         training_data='BSD300_color', 
                         training_noise=training_noise,
                        )

denoiser.eval()

# Choose a test dataset
# Note: grayscale/color of the test images must be consistent with grayscale/color of denoiser training data
grayscale='False'
if grayscale is True: 
    path = 'test_images/grayscale/'
else: 
    path = 'test_images/color/'
print('Test datasets: ', os.listdir(path))

test_folder = 'DIV2K'

# Directory to save the results
output_dir = '/home/jiayu/Desktop/Project/universal_inverse_problem-master/demosaicing/McMnoise/'
os.makedirs(output_dir, exist_ok=True)

# In solving linear inverse problems, in order to get good quality samples, beta should be small, that is lots of
# added noise in each iteration.
# This gives the algorithm the freedom to explore the space and arrive at a point on the manifold where the¶
# linear constraint is satisfied
results = []
im_num = 9

X = test_image(grayscale, path+test_folder+'/', im_num)

x = X.im

noise = denoising(x_size=x.size())
sample, interm_Ys = univ_inv_sol(denoiser, 
                                 x_c = noise.M_T(x), 
                                 task = noise,
                                 sig_0 = 1,
                                 sig_L = 0.01, 
                                 h0 = 0.01, 
                                 beta = 0.01,          
                                 freq = 40)
plot_all_samples(sample, interm_Ys)



# bayer = BayerPattern(x_size=x.size())
# # In solving linear inverse problems, in order to get good quality samples, beta should be small, that is lots of
# # added noise in each iteration.
# # This gives the algorithm the freedom to explore the space and arrive at a point on the manifold where the¶
# # linear constraint is satisfied
# sample, interm_Ys = univ_inv_sol(denoiser, 
#                                 x_c = bayer.M(x), 
#                                 task = bayer,
#                                 sig_0 = 1,
#                                 sig_L = 0.0008, 
#                                 h0 = 0.006, 
#                                 beta = 0.01,          
#                                 freq = 50)

save_image(sample, os.path.join(output_dir, f'bayer_{im_num}.tif'))


sample_image = to_pil_image(sample)


x = x.cpu()
sample = sample.cpu()

x = x.permute(1,2,0)
sample = sample.detach().permute(1,2,0)


ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True, channel_axis=-1, data_range=1.0) ,3)
psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy(), data_range=1.0),2)   
results.append({'image_number': im_num, 'psnr': psnr, 'ssim': ssim})

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(output_dir, 'results.csv')
results_df.to_csv(results_csv_path, index=False)    

print('Processing complete')    
