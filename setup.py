from setuptools import setup, find_packages

setup(
    name="pytorch_modular",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0, <3.0",
        "torchvision>=0.15",
        "matplotlib>=3.10",
        "requests>=2.32",
        "tqdm>=4.62",
        "numpy>=1.26",
    ]
)