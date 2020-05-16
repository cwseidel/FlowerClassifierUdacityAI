# AI Programming with Python
# Final Project Part 2
# 05/08/2020
# Chris Seidel
# This script contains support functions for
# train.py and predict.py

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
 

def image_sources(data_dir):
    ''' Prepare transforms and data loaders for training and testing a model
    '''
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    print(train_dir)
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    batch_size = 64

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    # train_data.class_to_idx
    # create the index to class mapping to attach to the model
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}

    return trainloader, testloader, idx_to_class


def create_model(arch, hidden_features):
    
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif arch == "vgg11_bn":
        model = models.vgg11_bn(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg13_bn":
        model = models.vgg13_bn(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
    else:
        print("architecture not supported.")
        exit()
    
    # turn off gradient back propagation
    for param in model.parameters():
        param.requires_grad = False

    if arch in ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn"]:
        in_features = model.classifier[0].in_features
    elif arch in ["densenet121","densenet161"]:
        in_features = model.classifier.in_features


    if hidden_features < 102 or hidden_features > in_features:
        print("please choose a number between {in_features} and 102")
        exit()
        
    classifier = nn.Sequential(nn.Linear(in_features, hidden_features),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_features, 102),
                           nn.LogSoftmax(dim=1))    
    
    model.classifier = classifier
    return(model)


def save_model(model, optimizer, file_name, idx_to_class, arch, hidden_units, epochs=0):
    checkpoint = {'epochs':epochs,
                  'arch':arch,
                  'hidden_units':hidden_units,
                  'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'idx_to_class':idx_to_class}
    torch.save(checkpoint, file_name)


def load_checkpoint(file_name):
    #checkpoint = torch.load(file_name)
    checkpoint = torch.load(file_name, map_location=lambda storage, loc:storage)
    model = create_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']

    # return a tuple containing the model, optimizer, and epochs
    return model, checkpoint['optimizer_state_dict'], checkpoint['epochs'] 

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    try:
        image = Image.open(image_path)
    except IOError as e:
        print(f"can not open image:{e}")
        return None
#    else:
#        return None
    
    w,h = image.size
    min_size = 256
    if w < min_size or h < min_size:
        print(f"image is too small: {w} x {h}")
        return None

    if w < h:
        scale_factor = w/min_size
    else:
        scale_factor = h/min_size

    # resize the image
    (width, height) = (int(image.width // scale_factor), int(image.height // scale_factor))
    im_rsz = image.resize((width, height))
   
    # crop the image
    crop_size = 224
    center_x = im_rsz.width // 2
    center_y = im_rsz.height // 2
    im_crop = im_rsz.crop((center_x - crop_size/2, center_y - crop_size/2, center_x + crop_size/2, center_y + crop_size/2))
    
    # convert to numpy array
    np_img = np.array(im_crop)

    # convert to 0-1 scale
    np_img = np_img/255
    # normalize to Library Values
    lib_mean_norm = np.array([0.485, 0.456, 0.406])
    lib_sd_norm = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - lib_mean_norm)/lib_sd_norm  
    # tranpose dimensions
    np_img_trans = np.transpose(np_img, (2,0,1))
    
    return np_img_trans
