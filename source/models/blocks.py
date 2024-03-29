import math
import numbers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F

############################## Custom functions ################################

def pla_sigmoid(x, k=1):
    return (2. / (1. + torch.exp(k * x)))


############################### Custom Modules #################################

def make_layer(block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)#nn.Sequential(*layers)


### Residual Blocks

class Residual_Block_Spec(nn.Module):
    def __init__(self, ch=256):
        super(Residual_Block_Spec, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.ReLU = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x

        output = self.ReLU(self.conv1(x))
        output = self.conv2(output)
        output = output * 0.1

        output = torch.add(output,identity_data)
        return output

### Residual in Residual Dense Blocks

def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, \
                bias=True, groups=1, norm_type=None, act_type='lrelu'):

    conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)

    block = [conv]

    if norm_type == 'batch':
        norm = nn.BatchNorm2d(out_nc)
        block.append(norm)
    if norm_type == 'instance':
        norm = nn.InstanceNorm2d(out_nc)
        block.append(norm)
    else:
        norm = None

    if act_type=='lrelu':
        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        block.append(act)
    elif act_type=='relu':
        act = nn.ReLU(inplace=False)
        block.append(act)
    else:
        act=None

    return nn.Sequential(*block)

class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias)

        self.conv5 = conv_block(nc+4*gc, nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class ResidualDenseBlock_3C(nn.Module):
    """
    Residual Dense Block
    style: 3 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, act_type='relu')
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, act_type='relu')
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, act_type='relu')

        self.conv4 = conv_block(nc+3*gc, nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4.mul(0.2) + x


### Upscaling blocks

class up_block(nn.Module):

    def __init__(self, in_ch, out_ch, up_factor=2, mode='nearest', kernel_size=3, stride=1, padding=1):
        super(up_block, self).__init__()

        if mode!='nearest' and mode != 'bilinear':
            raise NotImplementedError('mode [%s] is not implemented' % mode)
        else:
            self.mode = mode

        self.conv = conv_block(in_nc=in_ch, out_nc=out_ch, \
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, \
                    bias=False, groups=1, norm_type=None, act_type='lrelu')


    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)

        return x


### Extra

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
