#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='onnx2torch',
    version='1.5.15',
    install_requires=[
        'numpy>=1.16.4',
        'onnx>=1.9.0',
        'torch>=1.8.0',
        'torchvision>=0.9.0',
    ],
    packages=find_packages(),
)
