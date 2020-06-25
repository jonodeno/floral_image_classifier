# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. Develops an image classifier using PyTorch torchvision models and transfer learning to train a deep neural network on a set of floral images. Images are appropriately scaled, cropped and converted into PyTorch tensors to be ingested and analyzed by the deep neural network. 

## Requirements + Installation

Requires Python 3

To install the needed dependencies, run the following in the project directory:
```
$ pipenv install
```

## Usage

To train a model run the following in your project directory:
```
$ pipenv run python train.py flowers
```

You may specify the following arguments using the above command:
- `--save_dir`, the directory in which to save the checkpoint of your model
- `--arch`, You may select any of the models from [torchvision models from PyTorch](https://pytorch.org/docs/stable/torchvision/models.html). By typing it after `--arch` in the above command. The default is vgg16.
- `--learning_rate`, a number meant to scale the steps with which the model uses gradient descent
- `--hidden_units`, the number of hidden units used in the neural network of the classifier portion of the model. Can be a space separated list of numbers
- `--epochs`, the number of times that the model will train on the images in the `/flowers` directory.
- `--gpu`, specifies whether to run on the cpu or the gpu if gpu is available
