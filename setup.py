from setuptools import setup, find_packages

setup(
    name="pytorch_modular",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "requests",
        "tqdm",
        "numpy",
    ]
)