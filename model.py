import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, hidden_layers=[100], arch="vgg16", dropout=0.2):
        super().__init__()
        self.arch = arch
        # choose model based on arch
        if arch == "alexnet":
            self.model = models.alexnet(pretrained=True)
        elif arch == "densenet":
            self.model = models.densenet(pretrained=True)
        elif arch == "densenet121":
            self.model = models.densenet121(pretrained=True)
        elif arch == "densenet161":
            self.model = models.densenet169(pretrained=True)
        elif arch == "densenet201":
            self.model = models.densenet201(pretrained=True)
        elif arch == "inception":
            self.model = models.inception(pretrained=True)
        elif arch == "inception_v3":
            self.model = models.inception_v3(pretrained=True)
        elif arch == "resenet":
            self.model = models.resnet(pretrained=True)
        elif arch == "resnet101":
            self.model = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            self.model = models.resnet152(pretrained=True)
        elif arch == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            self.model = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif arch == "squeezenet":
            self.model = models.squeezenet(pretrained=True)
        elif arch == "squeezenet1_0":
            self.model = models.squeezenet1_0(pretrained=True)
        elif arch == "vgg":
            self.model = models.vgg(pretrained=True)
        elif arch == "vgg11":
            self.model = models.vgg11(pretrained=True)
        elif arch == "vgg11_bn":
            self.model = models.vgg_bn(pretrained=True)
        elif arch == "vgg13":
            self.model = models.vgg13(pretrained=True)
        elif arch == "vgg13_bn":
            self.model = models.vgg13_bn(pretrained=True)
        elif arch == "vgg16_bn":
            self.model = models.vgg16_bn(pretrained=True)
        elif arch == "vgg19":
            self.model = models.vgg19(pretrained=True)
        elif arch == "vgg19_bn":
            self.model = models.vgg19_bn(pretrained=True)
        else:
            self.model = models.vgg16(pretrained=True)
        
        self.model.arch = arch
        for param in self.model.parameters():
            param.requires_grad=False
        
        # build the classifier
        input_size = self.model.classifier[0].in_features
        self.layers = nn.ModuleList()
        layers = []
        layers.append(input_size)
        for layer in hidden_layers:
            layers.append(layer)            
        layer_sizes = zip(layers[:-1], layers[1:])

        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        classifier = nn.Sequential()
        iter_len = (len(layers)-1)*3
        for i in range(0,iter_len,3):
            classifier.add_module(str(i),self.layers[i//3])
            classifier.add_module(str(i+1),nn.ReLU())
            classifier.add_module(str(i+2),nn.Dropout(dropout))
        classifier.add_module("output",nn.Linear(hidden_layers[-1],102))
        classifier.add_module("log_output",nn.LogSoftmax(dim=1))
        self.model.classifier = classifier

            
            
        