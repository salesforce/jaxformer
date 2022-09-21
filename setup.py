from setuptools import setup, find_packages

setup(
  name = 'jaxformer',
  packages = find_packages(exclude=[]),
  version = '1.0.0',
  license='BSD3',
  description = 'Jaxformer',
  author = 'Erik Nijkamp',
  author_email = 'erik.nijkamp@gmail.com',
  url = 'https://github.com/salesforce/jaxformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
  ],
  install_requires=[
    'tensorflow-cpu==2.7.2'
  ],
)
