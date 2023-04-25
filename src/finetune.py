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
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

from models.model_manager import ModelManager

from dataset.dataset import TrainDatset, ValDatset

from utils.logger import Logger as Log
from utils.util import MetricLogger, SmoothedValue, NativeScaler
from utils.util import adjust_learning_rate, ensure_path, interpolate_pos_embed, intersectionAndUnionGPU
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


    def _set_dataloader(self):
        dataset_train = TrainDatset(args=self.args)
        dataset_val = ValDatset(args=self.args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True
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

        # shrink, shrink_linear, translator, expand_linear, expand
        checkpoint = torch.load(self.args.translator_weight, map_location='cpu')
        
        self._load_module_weight(self.model.shrink, 'shrink', checkpoint)
        self._load_module_weight(self.model.shrink_linear, 'shrink_linear', checkpoint)
        self._load_module_weight(self.model.translator, 'translator', checkpoint)
        self._load_module_weight(self.model.expand_linear, 'expand_linear', checkpoint)


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

        # freeze the mae encoder, shrink module parameters
        for name, param in self.model.named_parameters():
            if name[:3] == 'enc' or name[:6] == 'shrink':
                param.requires_grad = False

        self.model.to(self.device)

        # Log.info("\nModel = %s" % str(self.model))

        eff_batch_size = self.args.batch_size * self.args.accum_iter * self.args.eff_batch_adjust
        self.args.lr = self.args.blr * eff_batch_size / 256
        Log.info("base lr: %.2e" % (self.args.lr * 256 / eff_batch_size))
        Log.info("actual lr: %.2e" % self.args.lr)
        Log.info("accumulate grad iterations: %d" % self.args.accum_iter)
        Log.info("effective batch size: %d" % eff_batch_size)
        
        # data parallel
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])

        # set optimizer
        param_groups = add_weight_decay_ft(self.model, self.args.freeze_enc, weight_decay=self.args.weight_decay)
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
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
        self.optimizer.zero_grad()
        header = f'Epoch: [{epoch}]'

        for data_iter_step, (imgs, mask, catmask) in enumerate(metric_logger.log_every(self.train_loader, self.args.log_freq, header)):
        # for data_iter_step, (imgs, mask, catmask) in enumerate(self.train_loader):
            # per iteration lr scheduler
            if data_iter_step % self.args.accum_iter == 0:
                adjust_learning_rate(self.optimizer, data_iter_step / len(self.train_loader) + epoch, self.args)

            imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float)
            mask = mask.to(self.device, non_blocking=True, dtype=torch.float)
            catmask = catmask.to(self.device, non_blocking=True, dtype=torch.float)

            with torch.cuda.amp.autocast():
                loss, pred = self.model(imgs, catmask, mask, self.args.skip)

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

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # Log info of the whole epoch
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Epoch {epoch} done | Training time {total_time_str} | Averaged stats: " + str(metric_logger)
        msg += '\n\n'
        Log.info(msg)
    
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def validation(self):
        # add meta loop here

        # add metric here for evaluation
        # intersection, union, target = intersectionAndUnionGPU(pred.argmax(1), rearrange(mask, 'b t c h w -> (b t) c h w').argmax(1), 49, 255)

        # IoUb, IoUf = (intersection / (union + 1e-10)).cpu().numpy()  # mean of BG and FG  
        pass


    def finetune(self):
        Log.info("Start Finetuning")
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)

            if self.args.save_path and (epoch % self.args.save_freq == 0 or epoch + 1 == self.args.epochs):
                save_model_noddp(
                    args=self.args, model=self.model,
                    optimizer=self.optimizer,
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