# AI Programming with Python
# Final Project Part 2
# 05/08/2020
# Chris Seidel
# This script trains a classifier for a pre-trained network.
# The only required argument is a path to a directory structure of images
# e.g.: python train.py flowers


import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
#from PIL import Image
import numpy as np
#import json
# load all my functions
from support import create_model, image_sources, save_model, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a classifier for a pre-trained network.")
    parser.add_argument('data_dir', metavar='DATA_DIR', type=str,
                        help='The name of the data directory')
    parser.add_argument("--save_dir", "-s",
                        type=str,
                        default="checkpoint",
                        help="The name of a directory to save the checkpoint.",
                        required = False)
    parser.add_argument("--arch", "-ar",
                        type=str,
                        default="densenet121",
                        help="Pretrained Model architecture. The following are available: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, densenet121, densenet161",
                        required = False)
    parser.add_argument("--learning_rate", "-lr",
                        type=float,
                        default=0.001,
                        help="The learning rate",
                        required = False)
    parser.add_argument("--hidden_units", "-hu",
                        type=int,
                        default=512,
                        help="The number of units in the hidden layer.",
                        required=False)
    parser.add_argument("--gpu", "-g",
                        action="store_true",
#                        type=bool,                             
                        help="Just print the commands and exit.",
                        default = False,
                        required = False)
    parser.add_argument("--epochs", "-e",
                        type=int,
                        default=1,
                        help="The number of epochs for training.",
                        required = False)

    return parser.parse_args()


def create_checkpoint_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return False
    else:
        print ("Successfully created the directory %s " % path)
        return True

def main():
    # parse command line arguments                        
    args = parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    epochs = args.epochs
    arch = args.arch
    gpu = args.gpu
    learn_rate = args.learning_rate
    hidden_units = args.hidden_units
    past_epochs = 0
    
    # checkpoint filename
    file_name = save_dir + "/model_checkpoint.pth"

    print(f"learning_rate: {learn_rate}.")
    if learn_rate < 0 or learn_rate > 1:
        print("please choose a learning rate between 0 and 1.")
        exit()
    
    # make sure data_dir exists
    if os.path.isdir(data_dir):
        print(f"{data_dir} exists...")
    else:
        print(f"{data_dir} does not exist.")
        exit()

    # use gpu by default    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if user requests gpu but it's not available
    if device == "cpu" and gpu:
        print("sorry, gpu not available.")
        # exit()
    print(f"Device: {device}")
    
    # get image utilities and class mapping
    trainloader, testloader, idx_to_class = image_sources(data_dir)

    # check if save_dir exists    
    if not os.path.isdir(save_dir):
        # if not, create it
        if not create_checkpoint_dir(save_dir):
            exit()
        # create model
        print("Creating Model and Optimizer...")
        model = create_model(arch, hidden_units)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    else:
        model, optimizer_state_dict, past_epochs = load_checkpoint(file_name)
        print(f"model already trained with {past_epochs} epochs.")
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        optimizer.load_state_dict(optimizer_state_dict)
    
    # add inverted dictionary to model
    model.idx_to_class = idx_to_class

    #### train model ###
    # set criterion and optimizer
    criterion = nn.NLLLoss()
    model.to(device)

    n_batches = len(trainloader)
    print(f"Batches: {n_batches}")
    print_every = 2
    start_time = time.time()
    
    batch = 0
    running_loss = 0
    for e in range(epochs):
        for inputs, labels in trainloader:
            #if batch > 10:
            #    break
            batch += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # test the model
            if batch % print_every == 0:
                test_loss = 0
                accuracy = 0
                # turn off model update
                model.eval()
                with torch.no_grad():
                    #for i in np.arange(3):
                    #    inputs, labels = next(iter(testloader))
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"epoch:{e+1}/{epochs}..")
                    print(f"steps:{batch}/{n_batches}")
                    print(f"Test loss:{test_loss/len(testloader):.3f}")
                    print(f"Training loss:{running_loss/batch:.4f}")
                    print(f"Test accuracy:{100*accuracy/len(testloader):.3f}")
                    print(f"Elapsed Time:{time.time() - start_time}")
                model.train()
        else:
            print(f"epoch:{e+1}/{epochs}..")
            print(f"epoch time: {start_time/(e+1)}")
            print(f"Training loss:{running_loss/batch}")

    # save checkpoint
    epochs += past_epochs
    save_model(model, optimizer, file_name, idx_to_class, arch, hidden_units, epochs=epochs)

if __name__ == "__main__":
    main()
