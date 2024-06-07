import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

################################################# network class #################################################


class BF_CNN(nn.Module):

    def __init__(self, args):
        super(BF_CNN, self).__init__()

        self.padding = args.padding
        self.num_kernels = args.num_kernels
        self.kernel_size = args.kernel_size
        self.num_layers = args.num_layers
        self.num_channels = args.num_channels

        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])


        self.conv_layers.append(nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))
            self.running_sd.append( nn.Parameter(torch.ones(1,self.num_kernels,1,1), requires_grad=False) )
            g = (torch.randn( (1,self.num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True) )

        self.conv_layers.append(nn.Conv2d(self.num_kernels,self.num_channels, self.kernel_size, padding=self.padding , bias=False))


    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        x = relu(self.conv_layers[0](x))
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            # BF_BatchNorm
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)

            if self.conv_layers[l].training:
                x = x / sd_x.expand_as(x)
                self.running_sd[l-1].data = (1-.1) * self.running_sd[l-1].data + .1 * sd_x
                x = x * self.gammas[l-1].expand_as(x)

            else:
                x = x / self.running_sd[l-1].expand_as(x)
                x = x * self.gammas[l-1].expand_as(x)

            x = relu(x)

        x = self.conv_layers[-1](x)

        return x

################################################# network class #################################################
class UNet(nn.Module):
    
    def __init__(self, bias, residual_connection = False):
        
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,padding = 2, bias = bias)
        self.conv2 = nn.Conv2d(32,32,3,padding = 1, bias = bias)
        self.conv3 = nn.Conv2d(32,64,3,stride=2, padding = 1, bias = bias)
        self.conv4 = nn.Conv2d(64,64,3,padding = 1, bias=bias) 
        self.conv5 = nn.Conv2d(64,64,3,dilation=2, padding = 2, bias = bias)
        self.conv6 = nn.Conv2d(64,64,3,dilation = 4,padding = 4, bias = bias)
        self.conv7 = nn.ConvTranspose2d(64,64, 4,stride = 2, padding = 1, bias = bias)
        self.conv8 = nn.Conv2d(96,32,3,padding=1, bias = bias)
        self.conv9 = nn.Conv2d(32,1,5,padding = 2, bias = False)
        self.residual_connection = residual_connection
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--residual", action='store_true', help="use residual connection")
    @classmethod
    def build_model(cls, args):
        return cls(args.bias, args.residual)
  
    def forward(self, x):
        pad_right = x.shape[-2]%2
        pad_bottom = x.shape[-1]%2
        padding = nn.ZeroPad2d((0, pad_bottom,  0, pad_right))
        x = padding(x)

        out = F.relu(self.conv1(x))

        out_saved = F.relu(self.conv2(out))

        out = F.relu(self.conv3(out_saved))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = torch.cat([out,out_saved],dim = 1)
        out = F.relu(self.conv8(out))
        out = self.conv9(out)
        if self.residual_connection:
            out = x - out

        if pad_bottom:
            out = out[:, :, :, :-1]
        if pad_right:
            out = out[:, :, :-1, :]
        return out