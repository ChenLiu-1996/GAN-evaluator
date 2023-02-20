from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='gan-evaluator',
    version='1.13',
    license='MIT',
    author='Chen Liu',
    author_email='chen.liu.cl2482@yale.edu',
    packages={''},
    package_dir={'': 'src/utils/'},
    description='GAN Evaluator for IS and FID',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChenLiu-1996/GAN-evaluator',
    keywords='GAN, evaluator, IS, FID, inception',
    install_requires=['numpy', 'torch', 'torchvision', 'scipy', 'tqdm'],
)
