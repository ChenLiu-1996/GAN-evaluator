# Copyright: MIT Licence.
# Chen Liu (chen.liu.cl2482@yale.edu)
# https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator

import copy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from scipy.linalg import sqrtm
from scipy.stats import entropy
from torchvision.models.inception import inception_v3
from tqdm import tqdm


class GAN_Evaluator(object):
    """
    This evaluator computes the following metrics:
        - Inception Score (IS)
        - Frechet Inception Distance (FID)

    This evaluator will take in the real images and the fake/generated images.
    Then it will compute the activations from the real and fake images as well as the
    predictions from the fake images.
    The (fake) predictions will be used to compute IS, while
    the (real, fake) activations will be used to compute FID.
    If input image resolution < 75 x 75, we will upsample the image to accommodate Inception v3.

    The real and fake images can be provided to this evaluator in either of the following formats:
    1. dataloader
        `load_all_real_imgs`
        `load_all_fake_imgs`
    2. per-batch
        `fill_real_img_batch`
        `fill_fake_img_batch`

    !!! Please note: the latest IS and FID will be returned upon completion of either of the following:
        `load_all_fake_imgs`
        `fill_fake_img_batch`
    Return format:
        (IS mean, IS std, FID)
    *So please make sure you load real images before the fake images.*

    ---------
    Common Use Cases:
    1. For the purpose of on-the-fly evaluation during GAN training:
        We recommend pre-loading the real images using the dataloader format, and
        populate the fake images using the per-batch format as training goes on.
        - At the end of each epoch, you can clean the fake images using:
            `clear_fake_imgs`
        - In *unusual* cases where your real images change (such as in progressive growing GANs),
        you may want to clear the real images. You can do so via:
            `clear_real_imgs`
    2. For the purpose of offline evaluation of a saved dataset:
        We recommend pre-loading the real images and fake images.


    Partially inspired by:
    https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid
    https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    """

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_images_real: int = 1000,
                 num_images_fake: int = 1000,
                 IS_splits: int = 1) -> None:
        # NOTE: To pass in `num_images`, you can simply use `len(your_dataloader.dataset)`.

        self.min_resolution = 75  # Constrained by Inception v3.
        self.device = device

        # Set up dtype
        cuda = device.type == 'cuda'
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print(
                    "WARNING: You have a CUDA device, so you should probably set cuda=True"
                )
            self.dtype = torch.FloatTensor

        self.inception_encoder = inception_v3(weights='DEFAULT',
                                              transform_input=False).type(
                                                  self.dtype)
        self.inception_classifier = copy.deepcopy(self.inception_encoder.fc)
        self.inception_classifier.eval()
        self.inception_encoder.fc = torch.nn.Identity()
        self.inception_encoder.eval()
        self.upsample = torch.nn.Upsample(size=(299, 299),
                                          mode='bilinear').type(self.dtype)

        self.num_images_real = num_images_real
        self.num_images_fake = num_images_fake
        self.activation_vec_real = np.empty((self.num_images_real, 2048))
        self.activation_vec_fake = np.empty((self.num_images_fake, 2048))
        self.prediction_vec_fake = np.empty((self.num_images_fake, 1000))
        self.vec_real_pointer = 0
        self.vec_fake_pointer = 0

        # Parameter for IS computation. # of splits.
        self.IS_splits = IS_splits

    def fill_real_img_batch(self, real_batch: torch.Tensor) -> None:
        batch_size = real_batch.shape[0]
        real_batch = normalize(real_batch)
        real_batch = real_batch.type(torch.FloatTensor).to(self.device)

        _, _, H, W = real_batch.shape
        if H < self.min_resolution or W < self.min_resolution:
            real_batch = self.upsample(real_batch)

        # Get activations.
        with torch.no_grad():
            act_real = self.inception_encoder(real_batch)
        self.activation_vec_real[self.vec_real_pointer:self.vec_real_pointer +
                                 batch_size] = act_real.cpu().numpy()
        self.vec_real_pointer += batch_size
        return

    def fill_fake_img_batch(
            self,
            fake_batch: torch.Tensor,
            return_results: bool = True) -> Union[None, Tuple[float]]:
        batch_size = fake_batch.shape[0]
        fake_batch = normalize(fake_batch)
        fake_batch = fake_batch.type(torch.FloatTensor).to(self.device)

        _, _, H, W = fake_batch.shape
        if H < self.min_resolution or W < self.min_resolution:
            fake_batch = self.upsample(fake_batch)

        # Get activations and predictions.
        with torch.no_grad():
            act_fake = self.inception_encoder(fake_batch)
            pred_fake = self.inception_classifier(act_fake)
            prob_fake = torch.nn.functional.softmax(pred_fake,
                                                    dim=1).data.cpu().numpy()
        self.activation_vec_fake[self.vec_fake_pointer:self.vec_fake_pointer +
                                 batch_size] = act_fake.cpu().numpy()
        self.prediction_vec_fake[self.vec_fake_pointer:self.vec_fake_pointer +
                                 batch_size] = prob_fake
        self.vec_fake_pointer += batch_size

        if return_results:
            IS_mean, IS_std = self.compute_IS()
            FID = self.compute_FID()
            return IS_mean, IS_std, FID
        return

    def load_all_real_imgs(self,
                           real_loader: torch.utils.data.DataLoader,
                           idx_in_loader: Union[int, None] = None) -> None:
        for real_batch in tqdm(real_loader):
            if idx_in_loader is not None:
                real_batch = real_batch[idx_in_loader]
            self.fill_real_img_batch(real_batch)
        return

    def load_all_fake_imgs(
            self,
            fake_loader: torch.utils.data.DataLoader,
            img_idx_in_loader: Union[int, None] = None) -> Tuple[float]:
        for fake_batch in tqdm(fake_loader):
            if img_idx_in_loader is not None:
                fake_batch = fake_batch[img_idx_in_loader]
            self.fill_fake_img_batch(fake_batch, return_results=False)

        IS_mean, IS_std = self.compute_IS()
        FID = self.compute_FID()
        return IS_mean, IS_std, FID

    def clear_real_imgs(self) -> None:
        self.activation_vec_real = np.empty((self.num_images_real, 2048))
        self.vec_real_pointer = 0
        return

    def clear_fake_imgs(self) -> None:
        self.activation_vec_fake = np.empty((self.num_images_fake, 2048))
        self.prediction_vec_fake = np.empty((self.num_images_fake, 1000))
        self.vec_fake_pointer = 0
        return

    def compute_IS(self) -> Tuple[float]:
        # Compute the mean kl-div
        split_scores = []

        for k in range(self.IS_splits):
            subset = self.prediction_vec_fake[:self.vec_fake_pointer]
            part = subset[k *
                          (self.vec_fake_pointer // self.IS_splits):(k + 1) *
                          (self.vec_fake_pointer // self.IS_splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def compute_FID(self) -> float:
        subset_real = self.activation_vec_real[:self.vec_real_pointer]
        real_mean = np.mean(subset_real, axis=0)
        real_cov = np.cov(subset_real, rowvar=False)

        subset_fake = self.activation_vec_fake[:self.vec_fake_pointer]
        fake_mean = np.mean(subset_fake, axis=0)
        fake_cov = np.cov(subset_fake, rowvar=False)

        fid_value = frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        return fid_value


def normalize(
        image: Union[np.array, torch.Tensor],
        dynamic_range: List[float] = [-1, 1]) -> Union[np.array, torch.Tensor]:
    assert len(dynamic_range) == 2

    x1, x2 = image.min(), image.max()
    y1, y2 = dynamic_range[0], dynamic_range[1]

    slope = (y2 - y1) / (x2 - x1)
    offset = (y1 * x2 - y2 * x1) / (x2 - x1)

    image = image * slope + offset

    # Fix precision issue.
    image = image.clip(y1, y2)
    return image


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) -
            2 * tr_covmean)
