import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['big_sleep']
from version import __version__

setup(
  name = 'big-sleep',
  packages = find_packages(),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'dream = big_sleep.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Big Sleep',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/big-sleep',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text to image',
    'generative adversarial networks'
  ],
  install_requires=[
    'torch>=1.7.1',
    'einops>=0.3',
    'fire',
    'ftfy',
    'pytorch-pretrained-biggan',
    'regex',
    'torchvision>=0.8.2',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
