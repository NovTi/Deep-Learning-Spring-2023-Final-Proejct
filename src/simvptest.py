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

from models.simvp.model import SimVP_Model
from models.unet.unet import get_Unet

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
        dataset_test = TestDatset(args=self.args)
        dataset_valtest = ValTestDatset(args=self.args)
        self.valtest_loader = torch.utils.data.DataLoader(
            dataset_valtest, shuffle=False,
            batch_size=2,
            num_workers=self.args.num_workers,
            drop_last=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, shuffle=False,
            batch_size=2,
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
        checkpoint = torch.load(self.args.simvp_weight, map_location='cpu')
        # for lastest model of simvp
        checkpoint = checkpoint['state_dict']
        msg = self.model.load_state_dict(checkpoint, strict=True)
        Log.info('SimVP: ' + str(msg))

        model_dict = None
        checkpoint = None

        # load UNet weights
        checkpoint = torch.load(self.args.seghead_weight, map_location='cpu')['model']
        msg = self.seg.load_state_dict(checkpoint, strict=True)
        Log.info('Seg Head: ' + str(msg))


    def _set_model(self):
        self.model = SimVP_Model([11, 3, 160, 240])

        # small model
        # self.model = SimVP_Model(
        #     [11, 3, 160, 240],
        #     spatio_kernel_enc = 3,
        #     spatio_kernel_dec = 3,
        #     model_type = 'gSTA',
        #     hid_S = 32,
        #     hid_T = 256,
        #     N_T = 8,
        #     N_S = 4,
        # )

        self.seg = get_Unet()
        # load the SimVP and UNet weight
        self._load_weight()

        self.model.to(self.device)
        self.seg.to(self.device)
    
    def valtest(self):
        self.model.eval()
        self.seg.eval()
        start_time = time.time()

        pred_lst = []

        for i, (imgs, mask) in enumerate(self.valtest_loader):
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            if i % 50 == 0:
                Log.info(f"Currently Ep {i} | Total {len(self.valtest_loader)} Epochs")

            with torch.no_grad():
                # get the predict results
                x = self.model(imgs)[:, -1]  # only preserve the last frame
                # [B, 3, 64, 64]
                x = F.interpolate(x, size=(160, 240), mode='bilinear', align_corners=True)  # [B, 3, 224, 224]

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

    def test(self):
        self.model.eval()
        self.seg.eval()
        start_time = time.time()

        pred_lst = []

        for i, (imgs, path) in enumerate(self.test_loader):
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            if i % 50 == 0:
                Log.info(f"Currently Ep {i} | Total {len(self.test_loader)} Epochs")
            pdb.set_trace()
            with torch.no_grad():
                # get the predict results
                x = self.model(imgs)[:, -1]  # only preserve the last frame
                # [B, 3, 64, 64]
                x = F.interpolate(x, size=(160, 240), mode='bilinear', align_corners=True)  # [B, 3, 224, 224]

                # segment the predict frame
                x = self.seg(x)   # [B, 49, 160, 240]

                if i == 0:
                    total_pred = x.argmax(1).detach().cpu()
                else:
                    total_pred = torch.cat((total_pred, x.argmax(1).detach().cpu()), dim=0)

        np.save(os.path.join('results/', 'prediction.npy'), total_pred)
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
    # finetuner.valtest()
    finetuner.test()