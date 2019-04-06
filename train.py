import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

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

    # Using pre trained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
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
        # 
