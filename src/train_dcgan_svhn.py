# Copyright: MIT Licence.
# Chen Liu (chen.liu.cl2482@yale.edu)
# https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator

'''
Example: DCGAN on SVHN dataset.

This is just a simple example to show how to interface with our GAN_Evaluator.

The DCGAN architecture/training is adapted from the PyTorch DCGAN tutorial.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

import argparse
import os
import sys
from typing import List, Union

import numpy as np
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir)

from utils.attribute_hashmap import AttributeHashmap
from utils.eval_utils import GAN_Evaluator
from utils.log_utils import log
from utils.seed import seed_everything


class Generator(torch.nn.Module):

    def __init__(self, latent_dim: int = 128, num_channel: int = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.deconv1 = torch.nn.ConvTranspose2d(latent_dim, latent_dim * 8, 4,
                                                1, 0)
        self.deconv1_bn = torch.nn.BatchNorm2d(latent_dim * 8)
        self.deconv2 = torch.nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4,
                                                4, 2, 1)
        self.deconv2_bn = torch.nn.BatchNorm2d(latent_dim * 4)
        self.deconv3 = torch.nn.ConvTranspose2d(latent_dim * 4, latent_dim * 2,
                                                4, 2, 1)
        self.deconv3_bn = torch.nn.BatchNorm2d(latent_dim * 2)
        self.deconv4 = torch.nn.ConvTranspose2d(latent_dim * 2, latent_dim, 4,
                                                2, 1)
        self.deconv4_bn = torch.nn.BatchNorm2d(latent_dim)
        self.deconv5 = torch.nn.ConvTranspose2d(latent_dim, num_channel, 4, 2,
                                                1)

    def forward(self, input):
        x = torch.nn.functional.relu(self.deconv1_bn(self.deconv1(input)))
        x = torch.nn.functional.relu(self.deconv2_bn(self.deconv2(x)))
        x = torch.nn.functional.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.nn.functional.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x


class Discriminator(torch.nn.Module):

    def __init__(self, latent_dim: int = 128, num_channel: int = 3):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channel, latent_dim, 4, 2, 1)
        self.conv2 = torch.nn.Conv2d(latent_dim, latent_dim * 2, 4, 2, 1)
        self.conv2_bn = torch.nn.BatchNorm2d(latent_dim * 2)
        self.conv3 = torch.nn.Conv2d(latent_dim * 2, latent_dim * 4, 4, 2, 1)
        self.conv3_bn = torch.nn.BatchNorm2d(latent_dim * 4)
        self.conv4 = torch.nn.Conv2d(latent_dim * 4, latent_dim * 8, 4, 2, 1)
        self.conv4_bn = torch.nn.BatchNorm2d(latent_dim * 8)
        self.conv5 = torch.nn.Conv2d(latent_dim * 8, 1, 4, 1, 0)

    def forward(self, input):
        x = torch.nn.functional.leaky_relu(self.conv1(input), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


def normalize(
        image: Union[np.array, torch.Tensor],
        dynamic_range: List[float] = [0, 1]) -> Union[np.array, torch.Tensor]:
    assert len(dynamic_range) == 2

    x1, x2 = image.min(), image.max()
    y1, y2 = dynamic_range[0], dynamic_range[1]

    slope = (y2 - y1) / (x2 - x1)
    offset = (y1 * x2 - y2 * x1) / (x2 - x1)

    image = image * slope + offset

    # Fix precision issue.
    image = image.clip(y1, y2)
    return image


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    # Currently G and D are hard-coded for 64 x 64 resolution.
    target_imsize = 64

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(target_imsize),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ],
                                           p=0.8),
        torchvision.transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(
        config.dataset_dir,
        split='train',
        download=True,
        transform=transform_train),
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True)

    # Create folders.
    os.makedirs(config.plot_folder, exist_ok=True)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=config.log_dir, to_console=False)

    # Build the model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    opt_G = torch.optim.AdamW(generator.parameters(),
                              lr=config.learning_rate,
                              betas=(config.beta1, config.beta2))
    opt_D = torch.optim.AdamW(discriminator.parameters(),
                              lr=config.learning_rate,
                              betas=(config.beta1, config.beta2))

    loss_fn = torch.nn.BCELoss()

    # Our GAN Evaluator.
    evaluator = GAN_Evaluator(device=device,
                              num_images_real=len(train_loader.dataset),
                              num_images_fake=len(train_loader.dataset))

    # We can pre-load the real images in the format of a dataloader.
    # Of course you can do that in individual batches, but this way is neater.
    # Because in CIFAR10, each batch contains a (image, label) pair, we set `idx_in_loader` = 0.
    # If we only have images in the datalaoder, we can set `idx_in_loader` = None.
    evaluator.load_all_real_imgs(real_loader=train_loader, idx_in_loader=0)

    epoch_list, IS_list, FID_list = [], [], []
    for epoch_idx in range(config.max_epochs):
        generator.train()
        discriminator.train()
        num_visited = 0
        y_pred_real_sum, y_pred_fake_sum, loss_D_sum, loss_G_sum = 0, 0, 0, 0

        for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
            shall_plot = batch_idx % config.plot_interval == config.plot_interval - 1 or batch_idx == len(
                train_loader) - 1

            B = x.shape[0]
            x = normalize(x, dynamic_range=[-1, 1])
            num_visited += B

            ones = torch.ones(B, device=device)
            zeros = torch.zeros(B, device=device)
            x_real = x.type(torch.FloatTensor).to(device)

            # Update discriminator.
            z = torch.randn([B, generator.latent_dim, 1, 1], device=device)
            # Decouple from the generator's weights.
            with torch.no_grad():
                x_fake = generator(z)

            # Here comes the IS and FID values.
            # These are the values evaluated with the data available so far.
            # `IS_std` is only meaningful if `EVALUATOR.IS_splits` > 1.
            if shall_plot:
                IS_mean, IS_std, FID = evaluator.fill_fake_img_batch(
                    fake_batch=x_fake)
                epoch_list.append(epoch_idx + batch_idx / len(train_loader))
                IS_list.append(IS_mean)
                FID_list.append(FID)
            else:
                evaluator.fill_fake_img_batch(fake_batch=x_fake,
                                              return_results=False)

            y_pred_real = discriminator(x_real).view(-1)
            y_pred_fake = discriminator(x_fake).view(-1)

            loss_D = loss_fn(y_pred_real, ones) + loss_fn(y_pred_fake, zeros)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            y_pred_real_sum += torch.sum(y_pred_real)
            y_pred_fake_sum += torch.sum(y_pred_fake)
            loss_D_sum += loss_D.item() * B

            # Update generator.
            x_fake = generator(z)

            y_pred_fake = discriminator(x_fake).view(-1)
            loss_G = loss_fn(y_pred_fake, ones)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_G_sum += loss_G.item() * B

            # Plot intermediate results.
            if shall_plot:
                num_samples = 5
                rng = torch.Generator(device=device)
                rng.manual_seed(config.random_seed)
                H = W = target_imsize
                fig = plt.figure(figsize=(15, 6))
                for i in range(num_samples):
                    real_image = next(iter(train_loader))[0][0, ...][None, ...]
                    real_image = real_image.type(torch.FloatTensor).to(device)
                    with torch.no_grad():
                        discriminator.eval()
                        y_pred_real = discriminator(real_image).view(-1)
                        discriminator.train()
                    real_image = np.moveaxis(real_image.cpu().detach().numpy(),
                                             1, -1).reshape(H, W, 3)
                    real_image = normalize(real_image, dynamic_range=[0, 1])
                    ax = fig.add_subplot(2, num_samples, i + 1)
                    ax.imshow(real_image)
                    ax.set_axis_off()
                    ax.set_title('D(x): %.3f' % y_pred_real)
                    fix_z = torch.randn([1, generator.latent_dim, 1, 1],
                                        device=device,
                                        generator=rng)
                    with torch.no_grad():
                        generator.eval()
                        generated_image = generator(fix_z)
                        generator.train()
                        discriminator.eval()
                        y_pred_fake = discriminator(
                            generated_image.to(device)).view(-1)
                        discriminator.train()
                    generated_image = normalize(generated_image,
                                                dynamic_range=[0, 1])
                    ax = fig.add_subplot(2, num_samples, num_samples + i + 1)
                    ax.imshow(
                        np.moveaxis(generated_image.cpu().detach().numpy(), 1,
                                    -1).reshape(H, W, 3))
                    ax.set_title('D(G(z)): %.3f' % y_pred_fake)
                    ax.set_axis_off()

                plt.tight_layout()
                plt.savefig('%s/epoch_%s_batch_%s_generated' %
                            (config.plot_folder, str(epoch_idx).zfill(4),
                             str(batch_idx).zfill(4)))
                plt.close(fig=fig)

                log('Train [E %s/%s, B %s/%s] loss (G): %.3f, loss (D): %.3f, D(x): %.3f, D(G(z)): %.3f, IS: %.3f, FID: %.3f'
                    % (epoch_idx + 1, config.max_epochs, batch_idx + 1,
                       len(train_loader), loss_G_sum / num_visited,
                       loss_D_sum / num_visited, y_pred_real_sum / num_visited,
                       y_pred_fake_sum / num_visited, IS_mean, FID),
                    filepath=config.log_dir,
                    to_console=False)

        # Update the IS and FID curves every epoch.
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(epoch_list, IS_list, color='firebrick')
        ax.plot(epoch_list, IS_list, color='firebrick')
        ax.set_ylabel('Inception Score (IS)')
        ax.set_xlabel('Epoch')
        ax.spines[['right', 'top']].set_visible(False)
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(epoch_list, FID_list, color='firebrick')
        ax.plot(epoch_list, FID_list, color='firebrick')
        ax.set_ylabel('Frechet Inception Distance (FID)')
        ax.set_xlabel('Epoch')
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.savefig('%s/IS_FID_curve' % config.plot_folder)
        plt.close(fig=fig)

        # Need to clear up the fake images every epoch.
        evaluator.clear_fake_imgs()
    return


def parse_setting(config: AttributeHashmap) -> AttributeHashmap:
    if 'learning_rate' in config.keys():
        config.learning_rate = float(config.learning_rate)
    if 'beta1' in config.keys():
        config.beta1 = float(config.beta1)
    if 'beta2' in config.keys():
        config.beta2 = float(config.beta2)

    root = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in config.keys():
        if type(config[key]) == str and '$ROOT' in config[key]:
            config[key] = config[key].replace('$ROOT', root)

    config.log_dir = config.log_folder.rstrip('/') + '/' + \
        os.path.basename(
            config.config_file_name).replace('.yaml', '') + '_log.txt'

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entry point to train student network.')
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = parse_setting(AttributeHashmap(config))

    seed_everything(config.random_seed)
    train(config=config)
