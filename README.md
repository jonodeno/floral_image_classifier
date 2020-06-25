# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. Develops an image classifier using PyTorch torchvision models and transfer learning to train a deep neural network on a set of floral images. Images are appropriately scaled, cropped and converted into PyTorch tensors to be ingested and analyzed by the deep neural network. 

## Requirements + Installation

Requires Python 3

Clone the repo to your computer:
```
$ git clone https://github.com/jonodeno/floral_image_classifier.git
```

To install the needed dependencies, run the following in the project directory:
```
$ pipenv install
```

## Usage

### Training
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

Once you have completed training, the script will save your model as a checkpoint as `checkpoint.pth`

### Prediction
Once you have trained a model and have saved a checkpoint for that model, you may use the model for prediction on an image. Run the following command in your project directory:
```
$ pipenv run python predict.py {image_path} {checkpoint_path}
```

You may specify the following arguments using the above command:
- `--top_k`, by specifying a number, k, after this argument you specify the k most likely guesses for the image that you have supplied
- `--category_names`, use this to specify a different json file to use as the mapping of categories to names. By default, the script will use the supplied `cat_to_name.json`
- `--gpu`, specifies whether to run on the cpu or the gpu if gpu is available
