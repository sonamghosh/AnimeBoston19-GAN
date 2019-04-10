
"""
PyTorch StyleGAN implementation

Credit goes to Piotr Bialecki and Thomas Viehmann

and research credit goes to

T. Karras et. al. from Nvidia
A Style-Based Generator Architecture for Generative Adversarial Networks

Implementation is to generate Anime faced images from the Danbooru Anime face
dataset and pre-trained model on TensorFlow by Gwern

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle

import numpy as np

# Uncomment if using Jupyter Notebook
#import IPython


# Linear Layer
# inspired by H. Zhang et. al's following paper
# Fixup Initialization: Residual Learning without Normalization

class LinearLayer(nn.Module):
    # Linear layer with equalized learning rate and custom learning rate multiplier
    def __init__(self, in_dim, out_dim, gain=2**(0.5), use_wscale=False,
                 lrmul=1, bias=True):
        super(LinearLayer, self).__init__()
        he_std = gain * input_size**(-0.5)  # He init
        # equalized learning rate and custom learning rate multiplierself.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


# Convolution Layer
# Uses same trick as linear layer

class Conv2dLayer(nn.Module):
    # Conv layer with equalized learning rate and custom learning rate multiplier
    def __init__(self, in_channels, out_channels, kernel_size, gain=2**(0.5),
                 use_wscale=False, lrmul=1, bias=True,
                 intermediate=None, upscale=False):
        super(Conv2dLayer, self).__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (in_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGan, this is
            # incompatible with the non-fused way
            # Needs cleaning up
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # Applying a conv on W might be more efficient
            # But quadruples the weight (average)...
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size//2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size//2)

        if self.intermediate is not None:
            x = self.intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)

        return x


# Noise Layer
# Adds gaussian noise of learnable st dev (0 mean).
# Noise is per-pixel but constant over the channels

class NoiseLayer(nn.Module):
    # Adds noise per pixel (const over channels) with per channel weight
    def __init__(self, channels):
        super(NoiseLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # Trick: get all the noise layers and set each modules
            # .noise attribute, you can have pre-defined noise,
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x
