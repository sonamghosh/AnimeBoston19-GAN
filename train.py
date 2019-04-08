import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.autograd as autograd

from sagan import Generator, Discriminator
from utils import *

import argparse

def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyperparams
    parser.add_argument('--model', type=str, default='sagan',
                        choices=['sagan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp',
                        choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default='sagan_1')

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000,
                        help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # Instance noise
    # https://github.com/soumith/ganhacks/issues/14#issuecomment-312509518
    parser.add_argument('--inst_noise_sigma', type=float, default=0.1)
    parser.add_argument('--inst_noise_sigma_iters', type=int, default=200)

    # Using pre trained
    parser.add_argument('--pretrained_model', type=int, default=None)
    parser.add_argument('--state_dict_or_model', type=str, default=None,
                        help='Specify whether .pth pretrained model is a state_dict or a complete model')

    # Misc
    parser.add_argument('--manual_seed', type=int, default=29)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='cifar',
                        choices=['lsun', 'celeb', 'cifar', 'animefaces'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')
    parser.add_argument('--model_weights_dir', type=str, default='weights')
    parser.add_argument('--sample_images_dir', type=str, default='samples')

    # Step Size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    return parser.parse_args()


def build_tensorboard():
    from logger import Logger as lg
    logger = lg.Logger(config.log_path)


def save_sample(data_iter):
    real_images, _ = next(data_iter)
    save_image(denorm(real_images), os.path.join(config.sample_path, 'real.png'))


def reset_grad(D, G):
    D.optimizer.zero_grad()
    G.optimizer.zero_grad()


def compute_gradient_penalty(D_net, real_samples, real_labels, fake_samples):
    # Calculates gradient penalty loss for WGAN GP
    # Random weight term for interpolation between real and fake samples
    if torch.cuda.is_available():
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).expand_as(real_samples).cuda()
    else:
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).expand_as(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = torch.tensor(alpha * real_samples + (1 - alpha) * fake_samples, requires_grad=True)
    d_interpolates = D_net(interpolates)
    if torch.cuda.is_available():
        fake = torch.ones(d_interpolates.size()).cuda()
    else:
        fake = torch.ones(d_interpolates.size())
    # Get gradient w.r.t interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



if __name__ == "__main__":
    config = get_parameters()
    print(config)

    # Path
    log_path = os.path.join(config.log_path, config.version)
    sample_path = os.path.join(config.sample_path, config.version)
    model_save_path = os.path.join(config.model_save_path, config.version)

    if config.tensorboard():
        build_tensorboard()

    # Data iterator
    #data_iter = iter(data_loader)
    #step_per_epoch = len(data_loader)
    #model_save_step = int(config.model_save_step * step_per_epoch)


    # Start with trained model
    if config.pretrained_model:
        start = config.pretrained_model + 1
    else:
        start = 0

    # Start time
    start_time = time.time()

    # Build Model
    if torch.cuda.is_available():
        G_net = Generator(config.batch_size, config.imsize, config.z_dim,
                          config.g_conv_dim).cuda()
        D_net = Discriminator(config.batch_size, config.imsize,
                              config.d_conv_dim).cuda()
    else:
        G_net = Generator(config.batch_size, config.imsize, config.z_dim,
                          config.g_conv_dim)
        D_net = Discriminator(config.batch_size, config.imsize,
                              config.d_conv_dim)
    if config.parallel:
        G_net = nn.DataParallel(G_net)
        D_net = nn.DataParallel(D_net)

    # Loss and Optimizer
    g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G_net.parameters()), config.g_lr, [config.beta1, config.beta2])
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D_net.parameters()), config.d_lr, [config.beta1, config.beta2])

    c_loss = torch.nn.CrossEntropyLoss()

    # print networks
    print(G_net)
    print(D_net)

    # Training
    for step in range(start, config.total_step):
        # Train Discriminator
        D_net.train()
        G_net.train()

        try:
            real_images, _ = next(data_iter)
        except:
            data_iter = iter(data_loader)
            real_images, _ = next(data_iter)

        # Compute loss with real images
        # dr1, dr2m df1, df2, gf1, gf2 are attention scores
        real_images = tensor2var(real_images)
        d_out_real, dr1, dr2 = D_net(real_images)
        if config.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif config.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        # Apply gumbel softmax
        z = tensor2var(torch.rand(real_images.size(0), config.z_dim))
        fake_images, gf1, gf2 = G_net(z)
        d_out_fake, df1, df2 = D_net(fake_images)

        if config.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif config.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        # backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        if config.adv_loss == 'wgan-gp':
            d_loss_gp = compute_gradient_penalty(real_images, real_labels, fake_images)
            d_loss += config.lambda_gp + d_loss_gp

        reset_grad(D_net, G_net)
        d_loss.backward()
        d_optimizer.step()
