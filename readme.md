# Modularized PyTorch Utilities

The purpose of this repo is to avoid writing duplicate code for PyTorch development. We recommend installing the provided package `pytorch_modular` via pip:

```bash
pip install git+https://github.com/legendPerceptor/pytorch_modular.git
```

We currently provide 6 modules in this package.

## 1. Data setup

All data setup functionalities are located in `data_setup.py`.

We provide convenient functions for loading the whole [food_101_dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html). Users can load the whole dataset via the function `obtain_food101_dataset`. The `get_subset` function helps extract a subset of the data for faster model training/testing.

As the dataset is quite large, users can optionally use the function `directly_get_subset_of_food101_dataset` to download 20% of the data from a zipfile.

The `create_dataloaders` function returns a train dataloader and a test dataloader. Users need to provide it with the train image path and test image path.

## 2. Engine

The `engine.py` file provides training and testing functions.

The PyTorch training functions all follow similar steps, so it is possible and convenient to put the training loop into a function. We also provide a `train_step` and a `test_step` function to avoid writing a large train function. Most of the time, users only need to call the `train` function. It requires a model, a train dataloader, a test dataloader, an optimizer, a loss function, and the number of epochs with which you want to train the model.

## 3. Save the model

After the model is trained, it is important to save the model to a file. We provide a utility function `save_model` in `utils.py`.

## 4. Predictions

To use the trained model against some real-world data, we provide a `pred_and_plot_image` function for users to view the image and the prediction result.
