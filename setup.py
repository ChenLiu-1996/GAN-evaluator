from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent

setup(
    name='gan-evaluator',
    version='1.8',
    license='MIT',
    author="Chen Liu",
    author_email='chen.liu.cl2482@yale.edu',
    packages={''},
    package_dir={'': 'src/utils/'},
    description='GAN Evaluator for IS and FID',
    url='https://github.com/ChenLiu-1996/GAN-evaluator',
    keywords='GAN, evaluator, IS, FID',
    install_requires=['numpy', 'torch', 'torchvision', 'scipy', 'tqdm'],
)
