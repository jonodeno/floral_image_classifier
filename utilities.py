import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

"""
data = 
directory = directory where images are stored
nonrandom_transform = if True will use the nonrandom transforms from data_transforms
shuffle = if True will shuffle the images in the dataloader 
"""
def generate_loader(directory, nonrandom_transform=True, shuffle=False):
    batch_size = 20
    
    if nonrandom_transform:
        data = datasets.ImageFolder(directory, transforms['nonrandom'])
    else:
        data = datasets.ImageFolder(directory, transforms['random'])
        
    if shuffle:
        return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True), data
    else:
        return torch.utils.data.DataLoader(data, batch_size=batch_size), data

channel_means = [0.485, 0.456, 0.406]
channel_stdv = [0.229, 0.224, 0.255]

# random transforms to be used for training
# nonrandom to be used for testing and validation
transforms = {
    'random': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stdv)
    ]),
    'nonrandom': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stdv)
    ])
}

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    w, h = im.size
    
    if w > h:
        ratio = 256/float(h)
        im.thumbnail((int(w*ratio),256))
    else:
        ratio = 256/float(w)
        im.thumbnail((256,int(h*ratio)))
    
    # center is really at 256/2,256/2
    center = 256/2
    
    left = center - 224/2
    upper = center - 224/2
    right = center + 224/2
    lower = center + 224/2
    
    cropped = im.crop((left, upper, right, lower))
    
    np_image = np.array(cropped)/255
    
    stdv = [0.485, 0.456, 0.406]
    mean = [0.229, 0.224, 0.225]
    
    normalized = (np_image - mean)/stdv
    transposed = normalized.transpose(2,0,1)
    
    return transposed

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path[0])
    arch = checkpoint['arch']
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "densenet":
        model = models.densenet(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet169(pretrained=True)
    elif arch == "densenet201":
        model = models.densenet201(pretrained=True)
    elif arch == "inception":
        model = models.inception(pretrained=True)
    elif arch == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif arch == "resenet":
        model = models.resnet(pretrained=True)
    elif arch == "resnet101":
        model = models.resnet101(pretrained=True)
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
    elif arch == "squeezenet":
        model = models.squeezenet(pretrained=True)
    elif arch == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
    elif arch == "vgg":
        model = models.vgg(pretrained=True)
    elif arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif arch == "vgg11_bn":
        model = models.vgg_bn(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg13_bn":
        model = models.vgg13_bn(pretrained=True)
    elif arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    elif arch == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    return model