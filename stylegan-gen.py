
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


# Style Modification Layer
# In the generator, this layer is used after each non-affine instance norm layer
# The mean and variance is put back as an output of a linear layer that
# takes the latent style vector as inputs
# Adaptive Instance Norm (AdaIN)

class StyleModLayer(nn.Module):
    def __init__(self, latent_dim, channels, use_wscale):
        super(StyleModLayer, self).__init__()
        self.linear_layer = LinearLayer(latent_dim,
                                        channels * 2,
                                        gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear_layer(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


# Pixelnorm
# Normalizes per pixel across all channels
# Since default config uses the pixel norm in the g_mapping it forces
# the empirical st dev of the latent vector to 1

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNormLayer, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


# Upscale and blur Layers
# StyleGAN has two types of upscaling
# Regular one is a setting a block of 2x2 pixels to the value of the pixel
# to arrive an image that is scaled by 2
# Atlernative way is fused with convolution uses a stride 2 tranposed conv
# The generator blurs the layer by convolving with the simplest smoothing kernel

class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1)/2),
        )
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2dLayer(nn.Module):
    def __init__(self, factor=2, gain=1):
        super(Upscale2dLayer, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


# Generator Synthesis Blocks
# Each Block has two halfs
# Upscaling (if its  the first half) by a factor of two and blurring
#      - fused with convolution for the later layers.
# Convolution (if its the first half, having the channels for the later layers)
# Noise
# Activation (LeakyReLU in the reference model)
# Optionally Pixel norm (not used in the reference model)
# Instance Norm (optional, but used in reference model)
# The style modulation (i.e. setting the mean/st dev of the outputs after instance norm)

# Two of these sequences form a block that typically has
# out_channels = in_channels // 2 (in the earlier blocks, there are 512 input and 512 output channels)
# output_resolution = input_resolution * 2
# All of them are combined but the first two into a Module called Layer epilogue
# First block (4 x 4 ) pixels doesnt have an input
# The reesult of the first convolution is replaced by a trained constant.
# This is called the InputBlock, the others GSynthesisBlock.
# Nicer to do it the other way around where LayerEpilogue is the Layer and call conv from that

class LayerEpilogue(nn.Module):
    # Things to do at the end of each layer
    def __init__(self, channels, dlatent_size, use_wscale, use_noise,
                 use_pixel_norm, use_instance_norm, use_styles,
                 activation_layer):
        super(LayerEpilogue, self).__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.norm(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleModLayer(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale,
                 use_noise, use_pixel_norm, use_instance_norm,
                 use_styles, activation_layer):
        super(InputBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = LinearLayer(dlatent_size, nf*16, gain=gain/4, use_wscale=use_wscale)  # tweak gain to match offl implementation of prog GAN
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv = Conv2dLayer(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_pixel_norm,
                                  use_instance_norm, use_styles, activation_layer)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x
