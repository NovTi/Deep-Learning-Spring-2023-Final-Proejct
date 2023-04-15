import os
import sys
import pdb
import math
import time
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

from models.models_mae import ModelManager

from dataset.dataset import UnlabeledDataset



if __name__ == '__main__':
    vit = vit_base_patch16(
        img_size=224,
        num_classes=10,
        drop_path_rate=0.1
    )

    checkpoint = torch.load('results/2023-04-12:pretrain-day1/resize-wscale-work8/checkpoint-110.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']

    interpolate_pos_embed(vit, checkpoint_model)

    model_dict = vit.state_dict()

    for index, key in enumerate(model_dict.keys()):
        if model_dict[key].shape == checkpoint_model[key].shape:
            model_dict[key] = checkpoint_model[key]
        else:
            print( 'Pre-trained shape and model shape dismatch for {}'.format(key))

    pdb.set_trace()
    msg = vit.load_state_dict(model_dict, strict=True)

    a = 1