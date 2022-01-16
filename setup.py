from setuptools import setup, find_packages

setup(
    name='dataset_statistics',
    version='0.1',
    description='A project for extracting simple statistics from datasets with the nnUNet structure.',
    url='https://github.com/camgbus/dataset_statistics',
    keywords='python setuptools',
    packages=find_packages(include=['stat', 'stat.*']),
)