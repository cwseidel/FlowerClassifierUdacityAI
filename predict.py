# AI Programming with Python
# Final Project Part 2
# 05/08/2020
# Chris Seidel
# This script predicts a class for an image, based on a pretrained network.
# The only required arguments are the directory name of a checkpoint and a path to an image
# e.g.: python predict.py checkpoint image.jpg

import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
# load all my functions
from support import create_model, image_sources, load_checkpoint, process_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Read a targets file and align PE data with bwa.")
    parser.add_argument('checkpoint', metavar='checkpoint_dir', type=str,
                        help='The name of the checkpoint directory')

    parser.add_argument("path_to_image", metavar="path_to_image", type=str,
                        help="The path to an image file.")

    parser.add_argument("--gpu", "-g",
                        action="store_true",
                        help="Just print the commands and exit.",
                        default = False,
                        required = False)

    parser.add_argument("--top_k", "-k",
                        type=int,
                        default=1,
                        help="The number of top candidates to return",
                        required = False)

    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        default=False,
                        help="Report diagnostics",
                        required = False)
 
    return parser.parse_args()


def main():
    # parse command line arguments                        
    args = parse_args()
    checkpoint = args.checkpoint
    path_to_image = args.path_to_image
    top_k = args.top_k
    verbose = args.verbose
    gpu = args.gpu

    # load category to name mappings
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # use gpu by default    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if user requests gpu but it's not available
    if device == "cpu" and gpu:
        print("sorry, gpu not available.")
        # exit()
    if verbose:
        print(f"Device: {device}")
        
    # make sure checkpoint is an existing file
    checkpoint_file = checkpoint + "/model_checkpoint.pth"
    if os.path.isfile(checkpoint_file):
        if verbose:
            print(f"{checkpoint_file} exists...")
    else:
        print(f"{checkpoint_file} does not exist.")
        exit()

    # make sure image file exists
    if os.path.isfile(path_to_image):
        if verbose:
            print(f"{path_to_image} exists...")
    else:
        print(f"{path_to_image} does not exist.")
        exit()
    
    imp = process_image(path_to_image)
    if imp is None:
        print("Image is too small or can not be opened.")
        exit()
    
    model, optimizer_state_dict, epochs = load_checkpoint(checkpoint_file)
    criterion = nn.NLLLoss()
    model.to(device)
    model.eval()

    # convert to torch
    img_in = torch.from_numpy(imp)
    img_in = img_in.float()
    # add batch dimension
    img_in = img_in[None]
    # convert to correct data type
    #img_in = img_in.type(torch.cuda.FloatTensor)
    
    with torch.no_grad():
        img_in = img_in.to(device)
        logps = model.forward(img_in)
        ps = torch.exp(logps)
        #top_p, top_class = ps.topk(top_k, dim=1)
        top_p, top_class = ps.topk(top_k)
        # reduce dimension of classes
        c = top_class.squeeze()
        # convert to np array
        c = np.asarray(c)
        c = np.ndarray.tolist(c)
        # convert probabilities
        probs = np.asarray(top_p.squeeze())

    if top_k > 1:
        top_classes = [cat_to_name[model.idx_to_class[v]] for v in c]
        print(top_classes)
        print(probs)
    else:
        print(cat_to_name[model.idx_to_class[c]])
        print(f"probability: {probs}")

if __name__ == "__main__":
    main()
