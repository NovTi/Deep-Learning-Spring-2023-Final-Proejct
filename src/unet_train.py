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

from dataset.dataset import HRDatset

from utils.logger import Logger as Log
from utils.util import MetricLogger, SmoothedValue, NativeScaler
from utils.util import adjust_learning_rate, ensure_path
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, load_model_noddp, save_model_noddp, add_weight_decay_ft



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        output = F.softmax(h, dim =1)
        return h


def get_Unet(n_classes=49, load_weights=False):
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False, scale=1)
    net.outc = OutConv(64, n_classes)
    if load_weights:
        net.load_state_dict(torch.load("no_pretrain.pt"))
    return net


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device)
        self.loss_scaler = NativeScaler()
        # set dataloader
        self._set_dataloader()
        # set model, optimizer, scheduler
        self._set_model()

        # metric
        self.jaccard = JaccardIndex(task="multiclass", num_classes=49)


    def _set_dataloader(self):
        dataset_train = HRDatset(args=self.args, train=True)
        dataset_val = HRDatset(args=self.args, train=False)
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
            batch_size=self.args.batch_size
        )
    
    def _set_model(self):
        self.model = get_Unet()
        self.model.to(self.device)

        # Log.info("\nModel = %s" % str(self.model))

        eff_batch_size = self.args.batch_size * self.args.accum_iter * self.args.eff_batch_adjust
        self.args.lr = self.args.blr * eff_batch_size / 128   # used to be 256
        Log.info("base lr: %.2e" % (self.args.lr * 128 / eff_batch_size))
        Log.info("actual lr: %.2e" % self.args.lr)
        Log.info("accumulate grad iterations: %d" % self.args.accum_iter)
        Log.info("effective batch size: %d" % eff_batch_size)
        
        # data parallel
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])

        # get parameter groups
        param_groups = add_weight_decay_ft(
            self.model,
            weight_decay=self.args.weight_decay,
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


    def process_mask(self, mask, num_classes=49):
        num_frames, h, w = mask.shape
        new_mask = torch.zeros((num_frames, num_classes, h, w))
        for f in range(num_frames):
            # total cls num is 49
            for c in range(49):
                new_mask[f][c][torch.where(mask[f] == c)] = 1
        return new_mask


    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        # init the metric logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.optimizer.zero_grad()

        header = f'Epoch: [{epoch}]'
        for data_iter_step, (imgs, mask) in enumerate(metric_logger.log_every(self.train_loader, self.args.log_freq, header)):
            # per iteration lr scheduler
            if data_iter_step % self.args.accum_iter == 0:
                adjust_learning_rate(self.optimizer, data_iter_step / len(self.train_loader) + epoch, self.args)
            # convert to cuda
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)  # [B, 22, 3, 224, 224]

            imgs = rearrange(imgs, 'b t c h w->(b t) c h w')
            mask = rearrange(mask, 'b t h w->(b t) h w')
            
            # shuffle
            idx = torch.randperm(imgs.shape[0])
            imgs = imgs[idx]
            mask = mask[idx]

            # get the loss
            with torch.cuda.amp.autocast():
                imgs = self.model(imgs)   # pred: [(B T), 49, 160, 240]

            loss = F.cross_entropy(imgs, self.process_mask(mask).to(self.device))

            if epoch % self.args.val_log_freq == 0:
                if data_iter_step == 0:
                    total_pred = imgs.argmax(1).detach().cpu()
                    total_mask = mask
                else:
                    total_pred = torch.cat((total_pred, imgs.argmax(1).detach().cpu()), dim=0)
                    total_mask = torch.cat((total_mask, mask), dim=0)

            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                Log.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

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

        if epoch % self.args.val_log_freq == 0:
            # get mIoU for prediction
            miou = self.jaccard(total_pred, total_mask).item()
            total_pred = None
            total_mask = None

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if epoch % self.args.val_log_freq == 0:
            msg = f"Epoch {epoch} done | Training time {total_time_str} | mIoU {miou:.5f} | Averaged stats: " + str(metric_logger)
        else:
            msg = f"Epoch {epoch} done | Training time {total_time_str} | Averaged stats: " + str(metric_logger)

        msg += '\n\n'
        Log.info(msg)


    def validation(self, epoch):
        self.model.eval()
        start_time = time.time()

        for i, (imgs, mask) in enumerate(self.val_loader):
            # load images and masks
            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)  # [B, 3, 160, 240]

            imgs = rearrange(imgs, 'b t c h w->(b t) c h w')
            mask = rearrange(mask, 'b t h w->(b t) h w')
            # shuffle
            idx = torch.randperm(imgs.shape[0])
            imgs = imgs[idx]
            mask = mask[idx]

            # log information
            if i % self.args.val_log_freq == 0:
                Log.info(f"    Currently Ep {i} | Total {len(self.val_loader)} Epochs")

            with torch.no_grad():
                imgs = self.model(imgs)   # pred: [(B T), 49, 160, 240]

            if i == 0:
                total_pred = imgs.argmax(1).detach().cpu()
                total_mask = mask
            else:
                total_pred = torch.cat((total_pred, imgs.argmax(1).detach().cpu()), dim=0)
                total_mask = torch.cat((total_mask, mask), dim=0)

        # get mIoU for prediction
        miou = self.jaccard(total_pred, total_mask).item()
        total_pred = None
        total_mask = None

        # Log info of the validation
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"    Epoch {epoch} validating done | time {total_time_str} | 22-nd mIoU: {miou:.5f}"
        Log.info(msg)

        return miou


    def train(self):
        Log.info("Start Training")
        best_miou = 0.0
        best_epoch = 0
        for epoch in range(self.args.epochs):
            # train
            self.train_one_epoch(epoch)

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
    trainer = Trainer(args)
    trainer.train()