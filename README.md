# GAN Evaluator for Inception Score (IS) and Frechet Inception Distance (FID)

Chen Liu (chen.liu.cl2482@yale.edu)

Alex Wong (alex.wong@yale.edu)

## Main Contributions
1. We created a GAN evaluator for IS and FID that
    - is easy to use,
    - accepts data as either dataloaders or individual batches, and
    - supports on-the-fly evaluation during training.
2. We provided a simple demo script to demonstrate one common use case.

## Demo Script: Use [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial) to generate [SVHN](http://ufldl.stanford.edu/housenumbers/) numbers

The script can be found in [`src/train_dcgan_svhn.py`](https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator/blob/main/src/train_dcgan_svhn.py)

1. Real (top) and Generated (bottom) images after 50 epochs of training.
<img src = "debug_plot/dcgan_svhn/epoch_0049_batch_0286_generated.png" width=800>

2. IS and FID curves.
<img src = "debug_plot/dcgan_svhn/IS_FID_curve.png" width=800>

## The Evaluator for IS and FID
<details>
  <summary><b>Introduction to the Evaluator</b></summary>
<br>

More details can be found in [`src/utils/eval_utils.py/GAN_Evaluator`](https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator/blob/main/src/utils/eval_utils.py#L13).

```
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

COMMON USE CASES
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
```

</details>

## Repository Hierarchy
```
GAN-IS-FID-evaluator
    ├── config
    |   └── `dcgan_svhn.yaml`
    ├── data (*)
    ├── debug_plot (*)
    ├── logs (*)
    └── src
        ├── utils
        |   ├── `eval_utils.py`: THIS CONTAINS OUR `GAN_Evaluator`.
        |   └── other utility files.
        └── `train_dcgan_svhn.py`: our example use case.
```
Folders marked with (*), if not exist, will be created automatically when you run [`train_dcgan_svhn.py`](https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator/blob/main/src/train_dcgan_svhn.py).

## Usage
To run our example use case, do the following after activating the proper environment.
```
git clone git@github.com:ChenLiu-1996/GAN-IS-FID-evaluator.git
cd src
python train_dcgan_svhn.py --config ../config/dcgan_svhn.yaml
```

## Environement Setup
<details>
  <summary><b>Packages Needed</b></summary>
<br>

The `GAN_Evaluator` module itself only uses `numpy`, `scipy`, `torch`, `torchvision`, and (for aesthetics) `tqdm`.

To run the example script, it additionally requires `matplotlib`, `argparse`, and `yaml`.

</details>

<details>
  <summary><b>On our Yale Vision Lab server</b></summary>

- There is a virtualenv ready to use, located at
`/media/home/chliu/.virtualenv/mondi-image-gen/`.

- Alternatively, you can start from an existing environment "torch191-py38env",
and install the following packages:
```
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m pip install wget gdown numpy matplotlib pyyaml click scipy yacs scikit-learn
```

If you see error messages such as `Failed to build CUDA kernels for bias_act.`, you can fix it with:
```
python3 -m pip install ninja
```

</details>