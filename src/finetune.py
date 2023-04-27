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

from dataset.dataset import TrainDatset, ValDatset

from utils.logger import Logger as Log
from utils.util import MetricLogger, SmoothedValue, NativeScaler
from utils.util import adjust_learning_rate_ft, ensure_path, interpolate_pos_embed
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, load_model_noddp, save_model_noddp, add_weight_decay_ft


class Finetuner(object):
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device)
        self.loss_scaler = NativeScaler()
        self.ModelManager = ModelManager()
        # set dataloader
        self._set_dataloader()
        # set model, optimizer, scheduler
        self._set_model()

        # metric
        self.jaccard = JaccardIndex(task="multiclass", num_classes=49)


    def _set_dataloader(self):
        dataset_train = TrainDatset(args=self.args)
        self.dataset_val = ValDatset(args=self.args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset_val, shuffle=False,
            batch_size=1
        )


    def _get_translator_ckpt(self):
        checkpoint = torch.load(self.args.predict_weight, map_location='cpu')
        translator_weight = {}
        for key in checkpoint['model'].keys():
            if key[:3] != 'enc':
                translator_weight[key] = checkpoint['model'][key]
        path = '/'.join(self.args.predict_weight.split('/')[:-1])
        torch.save(translator_weight, os.path.join(path, 'translator.pth'))


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

        if not self.args.scratch:
            # shrink, shrink_linear, translator, expand_linear, expand
            # checkpoint = torch.load(self.args.translator_weight, map_location='cpu')
            checkpoint = torch.load('results/2023-04-26:finetune/ft_hr_nsc/best.pth', map_location='cpu')['model']
            
            # laod weight seperately to save memeory usage
            self._load_module_weight(self.model.shrink, 'shrink', checkpoint)
            self._load_module_weight(self.model.shrink_linear, 'shrink_linear', checkpoint)
            self._load_module_weight(self.model.translator, 'translator', checkpoint)
            self._load_module_weight(self.model.expand_linear, 'expand_linear', checkpoint)
            self._load_module_weight(self.model.dec, 'dec', checkpoint)
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
            device=self.args.device,
            learn_pos_embed=self.args.learn_pos_embed)
        # load the MAE encoder weight
        self._load_weight()

        # freeze the mae encoder, shrink, and shrink_linear module parameters
        for name, param in self.model.named_parameters():
            # notice here freezes both the shrink and shrink_linear module
            if name[:3] == 'enc' or name[:6] == 'shrink':
                param.requires_grad = False

        self.model.to(self.device)

        # Log.info("\nModel = %s" % str(self.model))

        eff_batch_size = self.args.batch_size * self.args.accum_iter * self.args.eff_batch_adjust
        self.args.lr = self.args.blr * eff_batch_size / 256   # used to be 256
        Log.info("base lr: %.2e" % (self.args.lr * 256 / eff_batch_size))
        Log.info("actual lr: %.2e" % self.args.lr)
        Log.info("accumulate grad iterations: %d" % self.args.accum_iter)
        Log.info("effective batch size: %d" % eff_batch_size)
        
        # data parallel
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])

        # get parameter groups
        param_groups = add_weight_decay_ft(
            self.model,
            module_ft=self.args.module_ft,
            weight_decay=self.args.weight_decay,
            scratch=self.args.scratch
        )
        # set optimizer
        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD(
                param_groups, lr=self.args.lr,
                momentum=self.args.momentum, nesterov=self.args.nesterov
            )
        elif self.args.opt == 'AdamW':
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))

        Log.info(f"Using {self.args.opt} optimizer")
        self.loss_scaler = NativeScaler()

        load_model_noddp(
            args=self.args,
            model=self.model,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler)


    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        # init the metric logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.optimizer.zero_grad()

        pred_lst = []
        mask_lst = []

        header = f'Epoch: [{epoch}]'
        for data_iter_step, (imgs, mask) in enumerate(metric_logger.log_every(self.train_loader, self.args.log_freq, header)):
            # per iteration lr scheduler
            if data_iter_step % self.args.accum_iter == 0:
                adjust_learning_rate_ft(self.optimizer, data_iter_step / len(self.train_loader) + epoch, self.args)
            # convert to cuda
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)  # [B, T, 3, 224, 224]
            mask = mask.to(self.device, non_blocking=True, dtype=torch.float)  # [B, 11, 160, 240]

            # get the loss
            with torch.cuda.amp.autocast():
                loss, pred = self.model(imgs, mask, self.args.skip)  # pred: last frame [B, 160, 240]

            mask_lst.append(mask[:, -1].detach().cpu())  # [B, 160, 240]
            pred_lst.append(pred.detach().cpu())    # [B, 160, 240]

            loss_value = loss.item()
            
            # if not math.isfinite(loss_value):
            #     Log.info("Loss is {}, stopping training".format(loss_value))
            #     sys.exit(1)

            loss /= self.args.accum_iter
            # loss.backward() here
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=(data_iter_step + 1) % self.args.accum_iter == 0)
            if (data_iter_step + 1) % self.args.accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        # concat all the predictions and masks
        pred_lst = torch.cat(pred_lst, dim=0)   # [1000, 160, 240]
        mask_lst = torch.cat(mask_lst, dim=0)   # [1000, 160, 240]

        # get mIoU for prediction
        miou = self.jaccard(pred_lst, mask_lst).item()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Epoch {epoch} done | Training time {total_time_str} | mIoU {miou:.5f} | Averaged stats: " + str(metric_logger)
        msg += '\n\n'
        Log.info(msg)


    def validation(self, epoch):
        self.model.eval()
        start_time = time.time()
         
        pred_last = []
        mask_last = []

        for i, (imgs, mask) in enumerate(self.val_loader):
            # load images and masks
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)  # [B, 11, 3, 224, 224]

            # add 22nd frame mask to list
            mask_last.append(mask[:, -1])   # [1, 160, 240]

            # log information
            if i % self.args.val_log_freq == 0:
                Log.info(f"    Currently Ep {i} | Total {len(self.val_loader)} Epochs")

            with torch.no_grad():
                # only preserve the 22nd frame
                x = self.model(imgs, label=None, skip=self.args.skip, train=False)  # [B, 49, 56, 56]
                # interpolate from 56x56 to 224x224
                x = F.interpolate(x, size=(160, 240), mode='bilinear', align_corners=True)  # [B, 49, 160, 240]
                pdb.set_trace()
                # add the predicted mask to list
                pred_last.append(x.argmax(1).detach().cpu())  # [B, 160, 240]

        # concat all the predictions and masks
        pred_last = torch.cat(pred_last, dim=0)   # [1000, 160, 240]
        mask_last = torch.cat(mask_last, dim=0)   # [1000, 160, 240]

        # get mIoU for prediction
        miou = self.jaccard(pred_last, mask_last).item()

        # Log info of the validation
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"    Epoch {epoch} validating done | time {total_time_str} | 22-nd mIoU: {miou:.5f}"
        Log.info(msg)

        return miou


    def finetune(self):
        Log.info("Start Finetuning")
        best_miou = 0.0
        best_epoch = 0
        for epoch in range(self.args.epochs):
            # train
            # self.train_one_epoch(epoch)

            if epoch % self.args.val_freq == 0:
                Log.info("Start Validating")
                current_miou = self.validation(epoch)

                if current_miou > best_miou:
                    best_miou = current_miou
                    best_epoch = epoch
                    Log.info(f"Saving model, current best: {best_miou:.5f}")
                    save_model_noddp(
                        args=self.args, model=self.model,
                        optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler, epoch=epoch, name='best')

            Log.info(f"Epoch {epoch} done | Best miou: {best_miou:.5f} | Best epoch {best_epoch}\n\n")
        # save the last epoch
        save_model_noddp(
            args=self.args, model=self.model,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler, epoch=epoch, name=f'last')



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
    # deal with skip
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
    finetuner = Finetuner(args)
    finetuner.finetune()