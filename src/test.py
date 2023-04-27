import os
import sys
import pdb
import copy
import math
import time
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from einops import rearrange
from torchmetrics import JaccardIndex

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

from models.model_manager import ModelManager

from dataset.dataset import TestDatset

from utils.logger import Logger as Log
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, load_model_noddp, ensure_path, interpolate_pos_embed


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.ModelManager = ModelManager()
        self.device = torch.device(args.device)
        # set dataloader
        self._set_dataloader()
        # set model, optimizer, scheduler
        self._set_model()


    def _set_dataloader(self):
        dataset_test = TestDatset(args=self.args)
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, shuffle=False,
            batch_size=10,
            num_workers=self.args.num_workers,
            drop_last=False
        )


    def _load_module_weight(self, module, name, checkpoint):
        model_dict = module.state_dict()
        for index, key in enumerate(model_dict.keys()):
            if model_dict[key].shape == checkpoint[name+'.'+key].shape:
                model_dict[key] = checkpoint[name+'.'+key]
            else:
                Log.info('Pre-trained shape and model shape dismatch for {}'.format(key))
                sys.exit(0)

        msg = module.load_state_dict(model_dict, strict=True)
        Log.info(f'{name}: ' + str(msg))
            

    def _load_weight(self):
        # load encoder weight from MAE
        checkpoint = torch.load(self.args.enc_weight, map_location='cpu')
        checkpoint = checkpoint['model']

        interpolate_pos_embed(self.model.enc, checkpoint)

        model_dict = self.model.enc.state_dict()

        for index, key in enumerate(model_dict.keys()):
            if model_dict[key].shape == checkpoint[key].shape:
                model_dict[key] = checkpoint[key]
            else:
                Log.info('Pre-trained shape and model shape dismatch for {}'.format(key))
                sys.exit(0)

        msg = self.model.enc.load_state_dict(model_dict, strict=True)
        Log.info('MAE Encoder: ' + str(msg))

        model_dict = None
        checkpoint = None

        checkpoint = torch.load(self.args.segmenter_weight, map_location='cpu')
        checkpoint = checkpoint['model']
        
        # load weight seperately to save the memory usage
        self._load_module_weight(self.model.shrink, 'shrink', checkpoint)
        self._load_module_weight(self.model.shrink_linear, 'shrink_linear', checkpoint)
        self._load_module_weight(self.model.translator, 'translator', checkpoint)
        self._load_module_weight(self.model.expand_linear, 'expand_linear', checkpoint)
        self._load_module_weight(self.model.f11_dec, 'f11_dec', checkpoint)
        self._load_module_weight(self.model.s11_dec, 's11_dec', checkpoint)
        self._load_module_weight(self.model.seghead, 'seghead', checkpoint)


    def _set_model(self):
        self.model = self.ModelManager.get_model(self.args.model)(
            args=self.args,
            img_size=self.args.input_size,
            shrink_embed=self.args.shrink_embed,
            trans_embed=self.args.trans_embed,
            num_heads=self.args.num_heads,
            mlp_ratio=self.args.mlp_ratio,
            num_layers=self.args.num_layers,
            device=self.args.device)
        # load the MAE encoder weight
        self._load_weight()

        self.model.to(self.device)


    def test(self):
        self.model.eval()
        start_time = time.time()

        resize_back = transforms.Resize((160, 240))
        pred_lst = []

        for i, (imgs) in enumerate(self.test_loader):
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            # if i % self.args.test_log_freq == 0:
            Log.info(f"Currently Ep {i} | Total {len(self.test_loader)} Epochs")

            with torch.no_grad():
                x = self.model(imgs, label=None, skip=self.args.skip, train=False)  # only preserve the last frame
                x = F.interpolate(x, size=224, mode='bilinear', align_corners=True)  # [B, 49, 224, 224]
                x = resize_back(x) # [B, 49, 160, 240]
                """ deal with the smooth value when resizing """
                x[torch.where(x>0.5)] = 1.0
                x[torch.where(x<0.5)] = 0.0
                pred_lst.append(x.argmax(1).cpu().numpy())  # [B, 160, 240]

        
        total_predict = np.concatenate(pred_lst, axis=0)
        np.save(os.path.join(self.args.save_path, 'prediction.npy'), total_predict)
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Testing done | time {total_time_str} | Good Luck"
        Log.info(msg)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='config file path')
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--exp_id', required=True, type=str, help='config modifications')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)
    # update exp_name and exp_id
    cfg['exp_name'] = args.exp_name
    cfg['exp_id'] = args.exp_id
    return cfg


if __name__ == "__main__":
    args = parse_config()
    seed = args.seed
    if seed is not None:
        cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deal with the scienciftic num
    args.min_lr = 1e-08
    if 'skip' in args.exp_id:
        args.skip = True
    else:
        args.skip = False

    # adjust device for easy testing on cpu environment
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    save_path = f"./results/{datetime.date.today()}:{args.exp_name}/{args.exp_id}"
    # save_path = f"./results/{args.exp_name}/{args.exp_id}"
    save_flag = ensure_path(save_path)
    args.save_path = save_path

    Log.init(
        log_file=os.path.join(save_path, 'output.log'),
        logfile_level='info',
        stdout_level='info',
        rewrite=True
    )

    # beautify the log output of the configuration
    msg = '\nConfig: \n'
    arg_lst = str(args).split('\n')
    for arg in arg_lst:
        msg += f'   {arg}\n'
    msg += f'\n[exp_name]: {args.exp_name}\n[exp_id]: {args.exp_id}\n[save_path]: {args.save_path}\n'
    Log.info(msg)

    args.update()

    # pretain
    finetuner = Tester(args)
    finetuner.test()