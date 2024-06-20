import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.fft
import gzip
import argparse
from network import BF_CNN, UNet

################################################# Helper Functions #################################################
def load_denoiser(architecture,grayscale, training_data, training_noise): 
    if architecture=='BF_CNN': 
        model = load_BF_CNN(grayscale, training_data, training_noise)
    elif architecture=='UNet':
        model = load_UNet(grayscale, training_data, training_noise)

        
    return model

def load_BF_CNN(grayscale, training_data, training_noise): 
    '''
    @ grayscale: if True, number of input and output channels are set to 1. Otherwise 3
    @ training_data: models provided in here have been trained on {BSD400, mnist, BSD300}
    @ training_noise: standard deviation of noise during training the denoiser
    '''
    parser = argparse.ArgumentParser(description='BF_CNN_color')
    parser.add_argument('--dir_name', default= '../noise_range_')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 20)
    if grayscale is True: 
        parser.add_argument('--num_channels', default= 1)
    else:
        parser.add_argument('--num_channels', default= 3)
    
    args = parser.parse_args('')

    model = BF_CNN(args)
    if torch.cuda.is_available():
        model = model.cuda()
    model_path = os.path.join('denoisers/BF_CNN',training_data,training_noise,'model.pt')
    if torch.cuda.is_available():
        learned_params =torch.load(model_path)

    else:
        learned_params =torch.load(model_path, map_location='cpu' )
    model.load_state_dict(learned_params)
    return model

def load_UNet(grayscale, training_data, training_noise, use_bias=False, use_residual=False): 
    
#     parser = argparse.ArgumentParser(description='UNet')
#     parser.add_argument('--dir_name', default= '../noise_range_')
#     parser.add_argument('--bias', type=bool, default=True)
#     parser.add_argument('--padding1', type=int, default=1)
#     parser.add_argument('--stride2', type=int, default=2)
#     parser.add_argument('--padding3', type=int, default=1)
#     parser.add_argument('--dilation4', type=int, default=2)
#     parser.add_argument('--padding4', type=int, default=2)
#     parser.add_argument('--dilation5', type=int, default=4)
#     parser.add_argument('--padding5', type=int, default=4)
#     parser.add_argument('--stride7', type=int, default=2)
#     parser.add_argument('--padding7', type=int, default=1)
#     parser.add_argument('--padding8', type=int, default=1)
#     parser.add_argument('--padding9', type=int, default=2)
#     parser.add_argument('--conv1.weight', default= 3)
#     parser.add_argument('--conv1.bias', default= False)
#     parser.add_argument('--conv2.weight', default= 3)
#     parser.add_argument('--conv2.bias', default= False)
#     parser.add_argument('--conv3.weight', default= 3)
#     parser.add_argument('--conv3.bias', default= False)
#     parser.add_argument('--conv4.weight', default= 3)
#     parser.add_argument('--conv4.bias', default= False)
#     parser.add_argument('--conv5.weight', default= 3)
#     parser.add_argument('--conv5.bias', default= False)
#     parser.add_argument('--conv6.weight', default= 3)
#     parser.add_argument('--conv6.bias', default= False)
#     parser.add_argument('--conv7.weight', default= 3)
#     parser.add_argument('--conv7.bias', default= False)
#     parser.add_argument('--conv8.weight', default= 3)
#     parser.add_argument('--conv8.bias', default= False)
#     parser.add_argument('--conv9.weight', default= 3)
#     parser.add_argument('--residual_connection', type=bool, default=False, help='Whether to use residual connections in the model')
#     if grayscale:
#         parser.add_argument('--image_channels', default=1)
#     else:
#         parser.add_argument('--image_channels', default=3)
#     args = parser.parse_args('')
#     model = UNet(args)
#     model_path = os.path.join('denoisers/UNet', training_data, training_noise, 'model.pt')
#     if torch.cuda.is_available():
#         learned_params =torch.load(model_path)

#     else:
#         learned_params =torch.load(model_path, map_location='cpu' )
#     model.load_state_dict(learned_params)




#     parser = argparse.ArgumentParser(description='Load UNet Model')
#     UNet.add_args(parser)
#     args = parser.parse_args([])
#     args.bias = use_bias
#     args.residual = use_residual
    
#     model = UNet.build_model(args)
    model = UNet(bias=use_bias, residual_connection=use_residual)
    model_path = os.path.join('denoisers/UNet', training_data, training_noise, 'model.pt')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in checkpoint and isinstance(checkpoint['model'], list) and len(checkpoint['model']) == 1:
        state_dict = checkpoint['model'][0]
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            print("Error: Expected a dictionary for the model state, but got a", type(state_dict))
            return None
    
#     if isinstance(data, list) and data:
#         state_dict = data[-1]
#         if isinstance(state_dict, dict):
#             model.load_state_dict(state_dict)
#         else:
#             raise TypeError("The last item in the list is not a dict.")
#     else:
#         raise TypeError("Loaded data is not list or is empty.")
        
    if torch.cuda.is_available():
        model.cuda()
        
    return model
        
   
    
#     if 'model' in checkpoint:
#         state_dict = checkpoint['model']
#         model.load_state_dict(state_dict)
#     else:
#         raise KeyError("No model state found in the checkpoint.")
        
#     if torch.cuda.is_available():
#         model.cuda()
        
#     return model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = os.path.join('denoisers/UNet', training_data, training_noise, 'model.pt')
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     return model

#     model = UNet(bias=False, residual_connection = False)
#     model_path = os.path.join('denoisers/UNet', training_data, training_noise, 'model.pt')
#     if torch.cuda.is_available():
#         learned_params =torch.load(model_path)

#     else:
#         learned_params =torch.load(model_path, map_location='cpu' )
#     model.load_state_dict(learned_params)
    
    
#     return model

#################################################
def single_image_loader(data_set_dire_path, image_number):
    
    if 'mnist' in data_set_dire_path.split('/'): 
        f = gzip.open(data_set_dire_path + '/t10k-images-idx3-ubyte.gz','r')
        f.read(16)
        buf = f.read(28 * 28 *10000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
        x = torch.tensor(data.reshape( 10000,28, 28).astype('float32'))[image_number:image_number+1]
        
    else: 
        all_names = os.listdir(data_set_dire_path)
        file_name = all_names[image_number]
        im = plt.imread(data_set_dire_path + file_name)
        if len(im.shape) == 3:
            x = torch.tensor(im).permute(2,0,1)
        elif len(im.shape) == 2:
            x = torch.tensor(im.reshape(1, im.shape[0], im.shape[1]))
            
    return x

class test_image: 
    def __init__(self, grayscale,path, image_num):
        super(test_image, self).__init__()
        
        self.grayscale = grayscale
        self.path = path
        self.image_num = image_num
        
        self.im = single_image_loader(self.path,self.image_num)
        if self.im.dtype == torch.uint8: 
            self.im = self.im/255
        if self.im.size()[0] == 3 and grayscale==True: 
            raise Exception('model is trained for grayscale images. Load a grayscale image')
        elif self.im.size()[0] == 1 and grayscale==False: 
            raise Exception('model is trained for color images. Load a color image')
        if torch.cuda.is_available():
            self.im = self.im.cuda()
        
    def show(self):
        if self.grayscale is True: 
            if torch.cuda.is_available():
                plt.imshow(self.im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else: 
                plt.imshow(self.im.squeeze(0), 'gray', vmin=0, vmax = 1)                
        else: 
            if torch.cuda.is_available():
                plt.imshow(self.im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else: 
                plt.imshow(self.im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('test image')
        plt.colorbar()
#         plt.axis('off');

    def crop(self, x0,y0,h,w):
        self.cropped_im = self.im[:, x0:x0+h, y0:y0+w]             
        if self.grayscale is True: 
            if torch.cuda.is_available():
                plt.imshow(self.cropped_im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else: 
                plt.imshow(self.cropped_im.squeeze(0), 'gray', vmin=0, vmax = 1)
                
        else: 
            if torch.cuda.is_available():            
                plt.imshow(self.cropped_im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else: 
                plt.imshow(self.cropped_im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('cropped test image')
        plt.colorbar()
#         plt.axis('off');        
        return self.cropped_im


#################################################
def rescale_image(im):
    if type(im) == torch.Tensor: 
        im = im.numpy()
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def plot_synthesis(intermed_Ys, sample):
    f, axs = plt.subplots(1,len(intermed_Ys), figsize = ( 4*len(intermed_Ys),4))
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()
            
        x = intermed_Ys[ax].permute(1,2,0).detach().numpy() 
        if x.shape[2] == 1: # if grayscale
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
        else: # if color
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()
        
    sample = sample.permute(1,2,0).detach().numpy()
    if sample.shape[2] == 1: # if grayscale
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else: # if color
        fig = axs[-1].imshow(rescale_image(sample))

    axs[-1].axis('off')
    print('value range', np.round(np.min(sample ),2), np.round(np.max(sample),2) )


def plot_sample(x, corrupted, sample):
    if torch.cuda.is_available():
        x = x.cpu()
        corrupted = corrupted.cpu()
        sample = sample.cpu()
        
    x = x.permute(1,2,0)
    corrupted = corrupted.permute(1,2,0)
    sample = sample.detach().permute(1,2,0)
        
    if x.size()!=corrupted.size():    
        h_diff = x.size()[0] - corrupted.size()[0]
        w_diff = x.size()[1] - corrupted.size()[1]
        x = x[0:x.size()[0]-h_diff,0:x.size()[1]-w_diff,: ]
        print('NOTE: psnr and ssim calculated using a cropped original image, because the original image is not divisible by the downsampling scale factor.')
        
    f, axs = plt.subplots(1,3, figsize = (15,5))
    axs = axs.ravel()        
    if x.shape[2] == 1: # if gray scale image
        fig = axs[0].imshow( x.squeeze(-1), 'gray', vmin=0, vmax = 1)
        axs[0].set_title('original')
        
        fig = axs[1].imshow(corrupted.squeeze(-1), 'gray',vmin=0, vmax = 1)
#         ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), corrupted.squeeze(-1).numpy()  ) ,3 )
#         psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() ),2)
        axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  
        
        fig = axs[2].imshow(sample.squeeze(-1),'gray' ,vmin=0, vmax = 1)
#         ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), sample.squeeze(-1).numpy()  ) ,3 )
#         psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy() ),2)
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );

            
    else: # if color image
        fig = axs[0].imshow( x, vmin=0, vmax = 1)
        axs[0].set_title('original')        
        
        fig = axs[1].imshow( torch.clip(corrupted,0,1), vmin=0, vmax = 1)
        #         ssim = structural_similarity(x, corrupted, channel_axis = 2)
        # ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True, channel_axis=-1) ,3 )
        ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True, channel_axis=-1, data_range=1.0) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy(), data_range=1.0),2)
        axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  

        fig = axs[2].imshow(torch.clip(sample, 0,1),vmin=0, vmax = 1)
        #         ssim = structural_similarity(x, sample, channel_axis = 2)
        # ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True, channel_axis=-1) ,3)
        ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True, channel_axis=-1, data_range=1.0) ,3)
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy(), data_range=1.0),2)   
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );

#         ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True, channel_axis=-1, data_range=1.0), 3)
#         psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy(), data_range=1.0),2)
#         axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  
        
#         fig = axs[2].imshow(torch.clip(sample, 0,1),vmin=0, vmax = 1)
#         ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True, channel_axis=-1, data_range=1.0) ,3)
#         psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy(), data_range=1.0),2)   
#         axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );
            
            
    for i in range(3): 
        axs[i].axis('off')
    
    


def plot_all_samples(sample, intermed_Ys):
    n_rows = int(np.ceil(len(intermed_Ys)/4))

    f, axs = plt.subplots(n_rows,4, figsize = ( 4*4, n_rows*4))
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()
            
        x = intermed_Ys[ax].detach().permute(1,2,0).numpy()
        if x.shape[2] == 1:
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
        else:
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')
    
    ### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()
        
    sample = sample.detach().permute(1,2,0).numpy()
    if sample.shape[2] == 1:
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else:
        fig = axs[-1].imshow(rescale_image(sample))
    axs[-1].axis('off')
    plt.colorbar(fig, ax=axs[-1], fraction=.05)


    for ax in range(len(intermed_Ys),n_rows*4 ):
        axs[ax].axis('off')


def plot_corrupted_im(x_c): 
    try:

        if torch.cuda.is_available():
            plt.imshow(x_c.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
        else: 
            plt.imshow(x_c.squeeze(0), 'gray', vmin=0, vmax = 1)
    except TypeError: 
        if torch.cuda.is_available():
            plt.imshow(x_c.permute(1,2,0).cpu(), vmin=0, vmax = 1)
        else: 
            plt.imshow(x_c.permute(1,2,0) , vmin=0, vmax = 1)

    plt.colorbar()   
    
    

    
    

def print_dim(measurment_dim, image_dim):
    print('*** Retained {} / {} ({}%) of dimensions'.format(int(measurment_dim), image_dim
                                                   , np.round(int(measurment_dim)/int(image_dim)*100,
                                                              decimals=3) ))    
    
###################################### Inverse problems Tasks ##################################
#############################################################################################
class synthesis:
    def __init__(self):
        super(synthesis, self).__init__()

    def M_T(self, x):
        return torch.zeros_like(x)

    def M(self, x):
        return torch.zeros_like(x)

class inpainting:
    '''
    makes a blanked area in the center
    @x_size : image size, tuple of (n_ch, im_d1,im_d2)
    @x0,y0: center of the blanked area
    @w: width of the blanked area
    @h: height of the blanked area
    '''
    def __init__(self, x_size,x0,y0,h, w):
        super(inpainting, self).__init__()

        n_ch , im_d1, im_d2 = x_size
        self.mask = torch.ones(x_size)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        c1, c2 = int(x0), int(y0)
        h , w= int(h/2), int(w/2)
        self.mask[0:n_ch, c1-h : c1+h , c2-w:c2+w] = 0

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask

    
class rand_pixels:
    '''
    @x_size : tuple of (n_ch, im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(rand_pixels, self).__init__()

        self.mask = np.zeros(x_size).flatten()
        self.mask[0:int(p*np.prod(x_size))] = 1
        self.mask = torch.tensor(np.random.choice(self.mask, size = x_size , replace = False).astype('float32').reshape(x_size))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()        
        
    def M_T(self, x):                                                                                                       
        return x*self.mask

    def M(self, x):
        return x*self.mask

    
class super_resolution:
    '''
    block averaging for super resolution.
    creates a low rank matrix (thin and tall) for down sampling
    @s: downsampling factor, int
    @x_size: tuple of three int  (n_ch, im_d1, im_d2)
    '''

    def __init__(self, x_size, s):
        super(super_resolution, self).__init__()

#         if x_size[1]%2 !=0 or x_size[2]%2 != 0 :
#             raise Exception("image dimensions need to be even")

        self.down_sampling_kernel = torch.ones(x_size[0],1,s,s)
        self.down_sampling_kernel = self.down_sampling_kernel/np.linalg.norm(self.down_sampling_kernel[0,0])
        if torch.cuda.is_available():
            self.down_sampling_kernel = self.down_sampling_kernel.cuda()
        self.x_size = x_size
        self.s = s

    def M_T(self, x):
        down_im = torch.nn.functional.conv2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])
        return down_im[0]

    def M(self, x):
        rec_im = torch.nn.functional.conv_transpose2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])

        return rec_im[0]


    
class random_basis:
    '''
    @x_size : tuple of (im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(random_basis, self).__init__()
        n_ch , im_d1, im_d2 = x_size
        self.x_size = x_size
        self.U, _ = torch.qr(torch.randn(int(np.prod(x_size)),int(np.prod(x_size)*p) ))
        if torch.cuda.is_available():
            self.U = self.U.cuda()

    def M_T(self, x):
        # gets 2d or 3d image and returns flatten partial measurement(1d)
        return torch.matmul(self.U.T,x.flatten())

    def M(self, x):
        # gets flatten partial measurement (1d), and returns 2d or 3d reconstruction
        return torch.matmul(self.U,x).reshape(self.x_size[0], self.x_size[1], self.x_size[2])


#### important: when using fftn from torch the reconstruction is more lossy than when fft2 from numpy
#### the difference between reconstruction and clean image in pytorch is of order of e-8, but in numpy is e-16

class spectral_super_resolution:
    '''
    creates a mask for dropping high frequency coefficients
    @im_d: dimension of the input image is (im_d, im_d)
    @p: portion of coefficients to keep
    '''
    def __init__(self, x_size, p):
        super(spectral_super_resolution, self).__init__()

        self.x_size = x_size
        
        xf = int(round(x_size[1]*np.sqrt(p) )/2)
        yf = int(round(x_size[1]*x_size[2]*p/xf )/4)
                
        mask = torch.ones((x_size[1],x_size[2]))

        mask[xf:x_size[1]-xf,:]=0
        mask[:, yf:x_size[2]-yf]=0        
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        # returns fft of each of the three color channels independently
        return self.mask*torch.fft.fftn(x, norm= 'ortho', dim = (1,2),s = (self.x_size[1], self.x_size[2]) )

    def M(self, x):
        return torch.real(torch.fft.ifftn(x, norm= 'ortho',  dim = (1,2), s = (self.x_size[1], self.x_size[2]) ))




class BayerPattern:
    def __init__(self, x_size):
        super(BayerPattern, self).__init__()

        n_ch, im_d1, im_d2 = x_size
        if n_ch != 3:
            raise ValueError("Image size must be in RGB format")

        self.mask = torch.zeros(x_size)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

  ### Fuji X-Trans Pattern
         ## R 
#         self.mask[0, 0::6, 4::6] = 1
#         self.mask[0, 1::6, 0::6] = 1
#         self.mask[0, 1::6, 2::6] = 1
#         self.mask[0, 2::6, 4::6] = 1
#         self.mask[0, 3::6, 1::6] = 1
#         self.mask[0, 4::6, 3::6] = 1
#         self.mask[0, 4::6, 5::6] = 1
#         self.mask[0, 5::6, 1::6] = 1

#         # G
#         self.mask[1, 0::6, 0::6] = 1
#         self.mask[1, 0::6, 2::6] = 1
#         self.mask[1, 0::6, 3::6] = 1
#         self.mask[1, 0::6, 5::6] = 1
#         self.mask[1, 1::6, 1::6] = 1
#         self.mask[1, 1::6, 4::6] = 1
#         self.mask[1, 2::6, 0::6] = 1
#         self.mask[1, 2::6, 2::6] = 1
#         self.mask[1, 2::6, 3::6] = 1
#         self.mask[1, 2::6, 5::6] = 1
#         self.mask[1, 3::6, 0::6] = 1
#         self.mask[1, 3::6, 2::6] = 1
#         self.mask[1, 3::6, 3::6] = 1
#         self.mask[1, 3::6, 5::6] = 1
#         self.mask[1, 4::6, 1::6] = 1
#         self.mask[1, 4::6, 4::6] = 1
#         self.mask[1, 5::6, 0::6] = 1
#         self.mask[1, 5::6, 2::6] = 1
#         self.mask[1, 5::6, 3::6] = 1
#         self.mask[1, 5::6, 5::6] = 1
#         # B 
#         self.mask[2, 0::6, 1::6] = 1
#         self.mask[2, 1::6, 3::6] = 1
#         self.mask[2, 1::6, 5::6] = 1
#         self.mask[2, 2::6, 1::6] = 1
#         self.mask[2, 3::6, 4::6] = 1
#         self.mask[2, 4::6, 0::6] = 1
#         self.mask[2, 4::6, 2::6] = 1
#         self.mask[2, 5::6, 4::6] = 1

#      ### RGBW #3 bayer pattern 
#         pattern_mask = torch.zeros((3, 4, 4))
#         # R
#         pattern_mask[0, 0, 2] = 1
#         pattern_mask[0, 2, 2] = 1
#         # G
#         pattern_mask[1, 0, 0] = 1
#         pattern_mask[1, 1, 2] = 1
#         pattern_mask[1, 2, 0] = 1
#         pattern_mask[1, 3, 2] = 1
#         # B
#         pattern_mask[2, 1, 0] = 1
#         pattern_mask[2, 3, 0] = 1
#         for ch in range(3):
#             self.mask[ch, :, :] = pattern_mask[ch, :, :].repeat(im_d1 // 4, im_d2 // 4)
#        # W
#         self.mask[:, :, 1::2] = 1

#    ### RGBW #2 bayer pattern 
#         pattern_mask = torch.zeros((3, 4, 4))
#         # R
#         pattern_mask[0, 0, 2] = 1
#         pattern_mask[0, 1, 2] = 1
#         # G
#         pattern_mask[1, 0, 0] = 1
#         pattern_mask[1, 1, 0] = 1
#         pattern_mask[1, 2, 2] = 1
#         pattern_mask[1, 3, 2] = 1
#         # B
#         pattern_mask[2, 2, 0] = 1
#         pattern_mask[2, 3, 0] = 1
#         for ch in range(3):
#             self.mask[ch, :, :] = pattern_mask[ch, :, :].repeat(im_d1 // 4, im_d2 // 4)
#        # W
#         self.mask[:, :, 1::2] = 1

#  ### RGBW #1 bayer pattern 
#         pattern_mask = torch.zeros((3, 4, 4))
#         # R
#         pattern_mask[0, 2, 3] = 1
#         pattern_mask[0, 3, 2] = 1
#         # G
#         pattern_mask[1, 0, 3] = 1
#         pattern_mask[1, 1, 2] = 1
#         pattern_mask[1, 2, 1] = 1
#         pattern_mask[1, 3, 0] = 1
#         # B
#         pattern_mask[2, 0, 1] = 1
#         pattern_mask[2, 1, 0] = 1
#         for ch in range(3):
#             self.mask[ch, :, :] = pattern_mask[ch, :, :].repeat(im_d1 // 4, im_d2 // 4)
#        # W
#         self.mask[:, 0::2, 0::2] = 1
#         self.mask[:, 1::2, 1::2] = 1

#  ### RGBW bayer pattern           
#         self.mask[0, ::2, 1::2] = 1  # red pixel
#         self.mask[:, ::2, ::2] = 1   # white pixel
#         self.mask[1, 1::2, 1::2] = 1 # green pixel
#         self.mask[2, 1::2, ::2] = 1  # blue pixel

 ## bayer pattern
        self.mask[0, ::2, 1::2] = 1  # red pixel
        self.mask[1, ::2, ::2] = 1   # green pixel
        self.mask[1, 1::2, 1::2] = 1 # green pixel
        self.mask[2, 1::2, ::2] = 1  # blue pixel

    def M_T(self, x):
        return x * self.mask

    def M(self, x):
        return x * self.mask


class joint:
    def __init__(self, x_size):
        super(joint, self).__init__()

        n_ch, im_d1, im_d2 = x_size
        if n_ch != 3:
            raise ValueError("Image size must be in RGB format")

        self.mask = torch.zeros(x_size)
        self.noise = np.random.normal(0, 25 / 255.0, x_size)
        self.noise = torch.from_numpy(self.noise).float()
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
            self.noise = self.noise.cuda()
            

        self.mask[0, ::2, 1::2] = 1  # red pixel
        self.mask[1, ::2, ::2] = 1   # green pixel
        self.mask[1, 1::2, 1::2] = 1 # green pixel
        self.mask[2, 1::2, ::2] = 1  # blue pixel
        
    def M_T(self, x):
#         noisy_bayer_image = x + self.noise
        bayer_image = x * self.mask
        noisy_bayer_image = bayer_image + self.noise
        return noisy_bayer_image


    def M(self, x):
#         noisy_bayer_image = x + self.noise
        bayer_image = x * self.mask
        noisy_bayer_image = bayer_image + self.noise
        return noisy_bayer_image
    

    
class joint_super_resolution:
    '''
    block averaging for super resolution.
    creates a low rank matrix (thin and tall) for down sampling
    @s: downsampling factor, int
    @x_size: tuple of three int  (n_ch, im_d1, im_d2)
    '''

    def __init__(self, x_size, s):
        super(joint_super_resolution, self).__init__()

#         if x_size[1]%2 !=0 or x_size[2]%2 != 0 :
#             raise Exception("image dimensions need to be even")

        self.down_sampling_kernel = torch.ones(x_size[0],1,s,s)
        self.down_sampling_kernel = self.down_sampling_kernel/np.linalg.norm(self.down_sampling_kernel[0,0])
        if torch.cuda.is_available():
            self.down_sampling_kernel = self.down_sampling_kernel.cuda()
        self.x_size = x_size
        self.s = s
        
        n_ch, im_d1, im_d2 = x_size
        self.mask = torch.zeros(n_ch, im_d1//4, im_d2//4)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
  
        self.mask[0, ::2, 1::2] = 1  # red pixel
        self.mask[1, ::2, ::2] = 1   # green pixel
        self.mask[1, 1::2, 1::2] = 1 # green pixel
        self.mask[2, 1::2, ::2] = 1  # blue pixel

        
        
        self.maskk = torch.zeros(x_size)
        if torch.cuda.is_available():
            self.maskk = self.maskk.cuda()
  
        self.maskk[0, ::2, 1::2] = 1  # red pixel
        self.maskk[1, ::2, ::2] = 1   # green pixel
        self.maskk[1, 1::2, 1::2] = 1 # green pixel
        self.maskk[2, 1::2, ::2] = 1  # blue pixel        

    def M_T(self, x):
        down_im = torch.nn.functional.conv2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])
        return down_im[0] * self.mask

    def M(self, x):
        rec_im = torch.nn.functional.conv_transpose2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])
        return rec_im * self.maskk
    
    
  


    
    
class JointBayerPattern:
    def __init__(self, x_size):
        super(JointBayerPattern, self).__init__()

        n_ch, im_d1, im_d2 = x_size
        if n_ch != 3:
            raise ValueError("Image size must be in RGB format")
            
        self.mask = torch.zeros(n_ch, im_d1, im_d2)    
#         self.mask = torch.zeros(n_ch, im_d1//s, im_d2//s)

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
 
 ## bayer pattern
        self.mask[0, ::2, 1::2] = 1  # red pixel
        self.mask[1, ::2, ::2] = 1   # green pixel
        self.mask[1, 1::2, 1::2] = 1 # green pixel
        self.mask[2, 1::2, ::2] = 1  # blue pixel
               

    def M_T(self, x):
        return x * self.mask

    def M(self, x):
        return x * self.mask


    
    
    
    

class sssuper_resolution:
    '''
    Extended for Bayer pattern demosaicing and up-sampling.
    @s: downsampling factor, int
    @x_size: tuple of three int  (n_ch, im_d1, im_d2)
    '''

    def __init__(self, x_size, s):
        super(sssuper_resolution, self).__init__()


        self.down_sampling_kernel = torch.ones(x_size[0], 1, s, s)
        self.down_sampling_kernel = self.down_sampling_kernel/np.linalg.norm(self.down_sampling_kernel[0,0])
        if torch.cuda.is_available():
            self.down_sampling_kernel = self.down_sampling_kernel.cuda()
        self.x_size = x_size
        self.s = s
        
        self.bayer_pattern = JointBayerPattern(x_size=(x_size[0], x_size[1]//s, x_size[2]//s))
#         self.bayer_pattern = JointBayerPattern(x_size, s)

    def M_T(self, x):
        # Apply down-sampling
        down_im = torch.nn.functional.conv2d(x.unsqueeze(0), self.down_sampling_kernel, stride=self.s, groups=x.size(0))
        # Apply Bayer pattern
        bayered_im = self.bayer_pattern.M(down_im[0])
        return bayered_im

    def M(self, x):
        # Reconstruct the full image size with up-sampling
        up_im = torch.nn.functional.conv_transpose2d(x.unsqueeze(0), self.down_sampling_kernel, stride=self.s, groups=x.size(0))
       
        # Apply the inverse Bayer to reconstruct RGB channels correctly
#         rec_im = self.bayer_pattern.M_T(up_im[0])
        return up_im

class denoising:
    def __init__(self, x_size):
        super(denoising, self).__init__()

        n_ch, im_d1, im_d2 = x_size
        if n_ch != 3:
            raise ValueError("Image size must be in RGB format")

        self.noise = np.random.normal(0, 25 / 255.0, x_size)
        self.noise = torch.from_numpy(self.noise).float()
        if torch.cuda.is_available():
            self.noise = self.noise.cuda()
            
           
    def M_T(self, x):
        noisy_bayer_image = x + self.noise
        return noisy_bayer_image


    def M(self, x):
        noisy_bayer_image = x + self.noise
        return noisy_bayer_image
