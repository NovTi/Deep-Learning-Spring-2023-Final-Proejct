# Reference: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py

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

from models.model_manager import ModelManager

from dataset.dataset import MAEUnlabeledDataset

from utils.logger import Logger as Log
from utils.util import MetricLogger, SmoothedValue, NativeScaler
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, load_model, save_model, adjust_learning_rate, ensure_path


class Pretrainer(object):
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device)
        self.loss_scaler = NativeScaler()
        self.ModelManager = ModelManager()
        # set dataloader
        self._set_dataloader()
        # set model, optimizer, scheduler
        self._set_model()

    def _set_dataloader(self):
        dataset_train = MAEUnlabeledDataset(args = self.args)
        a = dataset_train[0]
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

    def _set_model(self):
        self.model = self.ModelManager.get_model(self.args.model)(
            img_size = self.args.input_size,
            norm_pix_loss=self.args.norm_pix_loss)
        self.model.to(self.device)
        self.model_without_ddp = self.model
        Log.info("\nModel = %s" % str(self.model_without_ddp))

        eff_batch_size = self.args.batch_size * self.args.accum_iter
        self.args.lr = self.args.blr * eff_batch_size / 256
        Log.info("base lr: %.2e" % (self.args.lr * 256 / eff_batch_size))
        Log.info("actual lr: %.2e" % self.args.lr)
        Log.info("accumulate grad iterations: %d" % self.args.accum_iter)
        Log.info("effective batch size: %d" % eff_batch_size)
        
        # data parallel
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])
            self.model_without_ddp = model.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(self.model_without_ddp, self.args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))

        Log.info("Using AdamW optimizer")
        self.loss_scaler = NativeScaler()

        load_model(
            args=self.args,
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler)

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.optimizer.zero_grad()
        header = f'Epoch: [{epoch}]'
        for data_iter_step, samples in enumerate(metric_logger.log_every(self.train_loader, self.args.log_freq, header)):
            # per iteration lr scheduler
            if data_iter_step % self.args.accum_iter == 0:
                adjust_learning_rate(self.optimizer, data_iter_step / len(self.train_loader) + epoch, self.args)

            samples = samples.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, _, _ = self.model(samples, mask_ratio=self.args.mask_ratio)
                loss = torch.mean(loss)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                Log.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= self.args.accum_iter
            # contains loss.backward()
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=(data_iter_step + 1) % self.args.accum_iter == 0)
            if (data_iter_step + 1) % self.args.accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            # print the log info & reset the states.
            # if self.configer.get('iters') % self.configer.get('train', 'display_iter') == 0:
            #     Log.info('Ep {0} Iter {1} | loss {loss.val:.4f} (avg {loss.avg:.4f}) | '
            #              'lr {3} | time {batch_time.val:.2f}s/{2}iters'.format(
            #         self.configer.get('epoch'), self.configer.get('iters'), self.configer.get('train', 'display_iter'),
            #         ' '.join([f'{e:.4f}' for e in self.module_runner.get_lr(self.optimizer)]), batch_time=self.batch_time, loss=self.train_losses))
            #     self.batch_time.reset()
            #     self.train_losses.reset()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Epoch {epoch} done | Training time {total_time_str} | Averaged stats: " + str(metric_logger)
        msg += '\n\n'
        Log.info(msg)
    
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def pretrain(self):
        Log.info("Start Training")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_one_epoch(epoch)

            if self.args.save_path and (epoch % 5 == 0 or epoch + 1 == self.args.epochs):
                save_model(
                    args=self.args, model=self.model,
                    model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                    loss_scaler=self.loss_scaler, epoch=epoch)


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

    pretrainer = Pretrainer(args)
    pretrainer.pretrain()
