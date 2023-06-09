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

from models.cvitvp.cvitvp import CViT_VP
from models.unet.unet import get_Unet
from models.hrnet.hrnet import HRNet_W48

from dataset.dataset import TestDatset, ValTestDatset

from utils.logger import Logger as Log
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, load_model_noddp, ensure_path, interpolate_pos_embed


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        # set dataloader
        self._set_dataloader()
        # set model, optimizer, scheduler
        self._set_model()

        # metric
        self.jaccard = JaccardIndex(task="multiclass", num_classes=49)


    def _set_dataloader(self):
        # dataset_test = TestDatset(args=self.args)
        dataset_test = ValTestDatset(args=self.args)
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

        # load CViTVP weights
        checkpoint = torch.load(self.args.predict_weight, map_location='cpu')
        checkpoint = checkpoint['model']
        
        # load weight seperately to save the memory usage
        self._load_module_weight(self.model.shrink, 'shrink', checkpoint)
        self._load_module_weight(self.model.shrink_linear, 'shrink_linear', checkpoint)
        self._load_module_weight(self.model.translator, 'translator', checkpoint)
        self._load_module_weight(self.model.expand_linear, 'expand_linear', checkpoint)
        self._load_module_weight(self.model.dec, 'dec', checkpoint)

        checkpoint = None

        # # load HRNet weights
        # checkpoint = torch.load(self.args.seghead_weight, map_location='cpu')['model']
        # msg = self.seg.load_state_dict(checkpoint, strict=True)
        # Log.info('Seg Head: ' + str(msg))


    def _set_model(self):
        self.model = CViT_VP(
            img_size=224,
            shrink_embed=128,
            trans_embed=768,
            num_heads=12,
            mlp_ratio=6,
            num_layers=8,
            device='cuda',
            learn_pos_embed=False)

        self.seg = get_Unet()
        # load UNet weights
        checkpoint = torch.load(self.args.seghead_weight, map_location='cpu')['model']
        msg = self.seg.load_state_dict(checkpoint, strict=True)
        Log.info('Seg Head: ' + str(msg))
        
        # self.seg = HRNet_W48(self.args)
        # load the MAE encoder weight
        self._load_weight()

        self.model.to(self.device)
        self.seg.to(self.device)

    def val_test(self):
        self.model.eval()
        self.seg.eval()
        start_time = time.time()

        pred_lst = []

        for i, (imgs, mask) in enumerate(self.test_loader):
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            # if i % self.args.test_log_freq == 0:
            Log.info(f"Currently Ep {i} | Total {len(self.test_loader)} Epochs")

            with torch.no_grad():
                # get the predict results
                x = self.model(imgs, y=None, skip=self.args.skip, train=False)[:, -1]  # only preserve the last frame
                # [B, 3, 56, 56]
                # pdb.set_trace()
                x = F.interpolate(x, size=(160, 240), mode='bilinear', align_corners=True)  # [B, 3, 224, 224]
                # x = self.reverse_normalize(x)

                # segment the predict frame
                x = self.seg(x)   # [B, 49, 160, 240]

                if i == 0:
                    total_pred = x.argmax(1).detach().cpu()
                    total_mask = mask
                else:
                    total_pred = torch.cat((total_pred, x.argmax(1).detach().cpu()), dim=0)
                    total_mask = torch.cat((total_mask, mask), dim=0)

        miou = self.jaccard(total_pred, total_mask).item()        
        msg = f"miou {miou:.5f}"
        Log.info(msg)
        pdb.set_trace()
        # total_predict = np.concatenate(pred_lst, axis=0)
        # np.save(os.path.join(self.args.save_path, 'prediction.npy'), total_predict)
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Testing done | time {total_time_str} | Good Luck"
        Log.info(msg)


    def test(self):
        self.model.eval()
        self.seg.eval()
        start_time = time.time()

        pred_lst = []

        for i, (imgs, mask) in enumerate(self.test_loader):
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            # if i % self.args.test_log_freq == 0:
            Log.info(f"Currently Ep {i} | Total {len(self.test_loader)} Epochs")

            with torch.no_grad():
                # get the predict results
                x = self.model(imgs, y=None, skip=self.args.skip, train=False)[:, -1]  # only preserve the last frame
                # [B, 3, 56, 56]
                # pdb.set_trace()
                x = F.interpolate(x, size=(160, 240), mode='bilinear', align_corners=True)  # [B, 3, 224, 224]
                # x = self.reverse_normalize(x)

                # segment the predict frame
                x = self.seg(x)   # [B, 49, 160, 240]

                if i == 0:
                    total_pred = x.argmax(1).detach().cpu()
                    total_mask = mask
                else:
                    total_pred = torch.cat((total_pred, x.argmax(1).detach().cpu()), dim=0)
                    total_mask = torch.cat((total_mask, mask), dim=0)

        miou = self.jaccard(total_pred, total_mask).item()        
        msg = f"miou {miou:.5f}"
        Log.info(msg)
        pdb.set_trace()
        # total_predict = np.concatenate(pred_lst, axis=0)
        # np.save(os.path.join(self.args.save_path, 'prediction.npy'), total_predict)
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

    # pretain
    finetuner = Tester(args)
    finetuner.val_test()