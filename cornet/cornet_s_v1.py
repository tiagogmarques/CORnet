
import math
from collections import OrderedDict
from torch import nn
from V1_model import generate_filter_param, V1_model
import numpy as np

HASH = '1d3f7974'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class V1block(nn.Module):

    def __init__(self, in_channels, simple_channels, complex_channels, sf, theta, sigx, sigy, k_inh,
                 ksize=61, stride=4, ksize_div=13, div_type='gauss'):
        super().__init__()
        tot_channels = simple_channels + complex_channels
        self.v1_gfb = V1_model(in_channels,simple_channels=simple_channels,complex_channels=complex_channels,
                               ksize=ksize, div_type=div_type, stride=stride, ks_pool=ksize_div)
        k_exc = np.ones(tot_channels)
        spont = np.zeros(tot_channels)
        phase = np.zeros(tot_channels)
        self.v1_gfb.initialize(sf=sf, theta=theta, sigx_exc=sigx, sigy_exc=sigy, k_inh=k_inh,
                               phase=phase, k_exc=k_exc, spont=spont)

        for param in self.v1_gfb.parameters():
            param.requires_grad = False

    def forward(self, inp):
        return self.v1_gfb(inp)


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S_V1(seed=0):
    image_size = 224
    visual_degrees = 8

    rand_param = False
    inh_mult = 30
    fs = 0
    fc = 256

    nx, n_ratio, _, _, k_inh, theta, sf, _ = generate_filter_param(fs+fc, seed, rand_param)

    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = sigx * n_ratio
    k_inh = k_inh * inh_mult
    theta = theta/180 * np.pi

    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('gfb', V1block(3, 0, 4 * 64, sf=sf, theta=theta, sigx=sigx, sigy=sigy, k_inh=k_inh)),
            ('conv', nn.Conv2d(4 * 64, 64, kernel_size=1, stride=1, bias=False)),
            ('norm', nn.BatchNorm2d(4 * 64)),
            ('nonlin', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model