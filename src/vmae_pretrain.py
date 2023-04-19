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
from einops import rearrange

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory


from models.model_manager import ModelManager

from dataset.dataset import VMAEUnlabeledDataset

from optimizer.get_optim import create_optimizer

from utils.logger import Logger as Log
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, ensure_path

from utils.vmae_util import MetricLogger, SmoothedValue, auto_load_model, cosine_scheduler, save_model, NativeScaler


class Pretrainer(object):
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device)
        self.loss_scaler = NativeScaler()
        self.ModelManager = ModelManager()

        # set model, optimizer, scheduler, dataloader
        self._init_settings()

    def _init_settings(self):
        self.model = self.ModelManager.get_model(self.args.model)(
            drop_path_rate=self.args.drop_path,
            # drop_block_rate=None,
            decoder_depth=self.args.decoder_depth,
            use_checkpoint=self.args.use_checkpoint,
            num_frames=self.args.num_frames
        )

        patch_size = self.model.encoder.patch_embed.patch_size
        Log.info("Patch size = %s" % str(patch_size))
        self.args.window_size = (self.args.num_frames//2, self.args.input_size//patch_size[0], self.args.input_size//patch_size[1])
        self.args.patch_size = patch_size

        # set dataloader
        self._set_dataloader()

        self.model.to(self.device)
        self.model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        Log.info("\nModel = %s" % str(self.model_without_ddp))
        Log.info('Number of params: {} M\n'.format(n_parameters / 1e6))

        self.args.lr = self.args.lr * self.total_batch_size / 256
        self.args.min_lr = self.args.min_lr * self.total_batch_size / 256
        self.args.warmup_lr = self.args.warmup_lr * self.total_batch_size / 256
        Log.info("LR = %.8f" % args.lr)
        Log.info("Batch size = %d" % self.total_batch_size)
        Log.info("Number of training steps = %d" % self.num_training_steps_per_epoch)
        Log.info("Number of training examples per epoch = %d" % (self.total_batch_size * self.num_training_steps_per_epoch))

        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            self.model_without_ddp = model.module
        
        # set loss
        self.loss_func = nn.MSELoss()

        # set the optimizer
        self.optimizer = create_optimizer(self.args, self.model_without_ddp)
        self.loss_scaler = NativeScaler()
        Log.info(f"Using {self.args.opt} optimizer")

        # set scheduler
        Log.info("Use step level LR & WD scheduler!")
        self.lr_schedule_values = cosine_scheduler(
            self.args.lr, self.args.min_lr, self.args.epochs, self.num_training_steps_per_epoch,
            warmup_epochs=self.args.warmup_epochs, warmup_steps=self.args.warmup_steps,
        )
        if self.args.weight_decay_end is None:
            self.args.weight_decay_end = self.args.weight_decay
        self.wd_schedule_values = cosine_scheduler(
            self.args.weight_decay, self.args.weight_decay_end, self.args.epochs, self.num_training_steps_per_epoch)
        Log.info("Max WD = %.7f, Min WD = %.7f" % (max(self.wd_schedule_values), min(self.wd_schedule_values)))

        auto_load_model(
            args=self.args,
            model=self.model, 
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler)

    def _set_dataloader(self):
        dataset_train = VMAEUnlabeledDataset(args=self.args)
        # a = dataset_train[0]
        # not distributed
        num_tasks = 1
        global_rank = 0

        self.total_batch_size = self.args.batch_size * num_tasks
        self.num_training_steps_per_epoch = len(dataset_train) // self.total_batch_size

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

    def train_one_epoch(self, epoch, max_norm: float=0):
        self.model.train()
        start_time = time.time()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        start_steps = epoch * self.num_training_steps_per_epoch

        self.optimizer.zero_grad()
        header = f'Epoch: [{epoch}]'
        for step, batch in enumerate(metric_logger.log_every(self.train_loader, self.args.log_freq, header)):
            it = start_steps + step  # global training iteration
            # adjust lr
            if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if self.lr_schedule_values is not None:
                        param_group["lr"] = self.lr_schedule_values[it] * param_group["lr_scale"]
                    if self.wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = self.wd_schedule_values[it]

            videos, bool_masked_pos = batch
            videos = videos.to(self.args.device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(self.args.device, non_blocking=True).flatten(1).to(torch.bool)

            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor([0.485, 0.456, 0.406]).to(self.args.device)[None, :, None, None, None]
                std = torch.as_tensor([0.229, 0.224, 0.225]).to(self.args.device)[None, :, None, None, None]
                unnorm_videos = videos * std + mean  # in [0, 1]

                if self.args.normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', \
                                                p0=2, p1=self.args.patch_size[0], p2=self.args.patch_size[0])
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', \
                                                p0=2, p1=self.args.patch_size[0], p2=self.args.patch_size[0])

                B, _, C = videos_patch.shape
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

            with torch.cuda.amp.autocast():
                outputs = self.model(videos, bool_masked_pos)
                loss = self.loss_func(input=outputs, target=labels)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                Log.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            self.optimizer.zero_grad()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=max_norm,
                                    parameters=self.model.parameters(), create_graph=is_second_order)
            loss_scale_value = self.loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in self.optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in self.optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

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
        torch.cuda.empty_cache()
        start_time = time.time()
        Log.info("Start Training")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_one_epoch(epoch=epoch)

            if self.args.save_path and (epoch % self.args.save_freq == 0 or epoch + 1 == self.args.epochs):
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

    save_path = f"./results/{datetime.date.today()}:{args.exp_name}/{args.exp_id}"
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

    # pretrain
    pretrainer = Pretrainer(args)
    pretrainer.pretrain()
