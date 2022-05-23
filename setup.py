import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['Trachea']
#from version import __version__

setup(
  name = 'Trachea',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'making computer listen to you',
  author = 'Shauray Singh',
  author_email = 'shauraysingh08@gmail.com',
  url = 'https://github.com/shauray8/Trachea',
  keywords = ['deep learning',"listen", 'machine learning'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'torchaudio',
      'pillow',
      "opencv2",
      "tensorboard".
  ],
)
