
import os
import pdb
import importlib
from addict import Dict

from src.tools.utils import path2module, log_uniform

class Configer(object):

    def __init__(self, parser):
        self.args = parser.__dict__
        if not os.path.exists(parser.config):
            raise ValueError(f"Config path error: {parser.config}")
        self.config = Dict(importlib.import_module(path2module(parser.config)).CONFIG)
        for k, v in self.args.items():
            self.config[k] = v
        self.modified_config = self.modify_config()

    def __getattr__(self, key):
        return self.config[key]

    def exists(self, *keys):
        item = self.config
        for i in range(len(keys)):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(f"Invalid key: {keys}")
        return not (isinstance(item, Dict) and len(item) == 0)

    def get(self, *keys):
        item = self.config
        for i in range(len(keys)):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(keys)
        if isinstance(item, Dict) and len(item) == 0:
            return None
        return item
    
    def set(self, keys, val):
        item = self.config
        for i in range(len(keys) - 1):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(f"Invalid key: {keys}")
        assert isinstance(item, Dict)
        item[keys[-1]] = val

    def update(self, to_modify):
        for k, v in to_modify.items():
            self.set(k, v)

    def plus_one(self, *keys):
        val = self.get(*keys)
        assert isinstance(val, int)
        self.set(keys, val+1)
    
    def log_config(self):
        msg = ['\n\nConfig:\n']
        for k, v in self.config.items():
            if isinstance(v, Dict):
                msg.append(f'[{k}]:\n')
                for kk, vv in v.items():
                    msg.append(f'\t[{kk}]: {vv}\n')
            else:
                msg.append(f'[{k}]: {v}\n')
        self.logger.info(''.join(msg))
    
    def log_modified_config(self):
        msg = ['\n\nModified Config:\n']
        for k, v in self.modified_config.items():
            msg.append(f'{k}: {v}\n')
        self.logger.info(''.join(msg))

    def modify_config(self):
        to_modify = {}
        exp_name, exp_id = self.args['exp_name'], self.args['exp_id']
        if self.config['debug']:
            to_modify.update(DEBUG_CONFIG)

        if exp_name == 'dlv3_r50_ctr':
            keys = exp_id.split('_')
            val_dict = {'eps': True, 'std': False}
            ctr_dict = {'ctr': 'contrast_ce_loss', 'ce': 'ce_loss'}
            to_modify.update({
                ('val', 'episodic_val'): val_dict[keys[0]], # eps | std
                ('loss', 'loss_type'): ctr_dict[keys[1]],   # ctr | ce
                ('train_split', ): int(keys[2])             # 0 | 1 | 2 | 3
            })
        elif exp_name == 'dlv3_r50_ctr_rand':
            to_modify.update({
                ('val', 'episodic_val'): True,
                ('loss', 'loss_type'): 'contrast_ce_loss',
                ('contrast', 'loss_weight'): log_uniform(0.05, 0.5),
                ('train_split', ): int(exp_id[0])           # 0 | 1 | 2 | 3
            })
        elif exp_name == 'ft_psp_r50_ce' or exp_name == 'ft_psp_r50_ce_test' :
            if exp_id == 'modify_test':  # not real "test", but test whether the code can run
                to_modify.update({
                ('val', 'episodic_val'): True,
                ('loss', 'loss_type'): 'ce_loss',
                ('network', 'model_name'): 'pspnet_contrast',
                ('finetune', 'bottle_conv'): 1,
                ('finetune', 'bottle_lr_time'): 0.1,  #  eg. 0.1 / 0.3
                ('finetune', 'bottle_ft'): True,   # if lr times is 00, not finetune
                ('finetune', 'bottle_type'): 0,  # 0 | 1
                ('train_split', ): 0,          # 0 | 1 | 2 | 3
                ('shot', ): 1
            })
            else:
                # c1_01_bt1_sp0_sh1  conv 1x1, lr times 0.1,  bottleneck type: 1, split 0, shot 1
                keys = exp_id.split('_')    # ['c1', '01', 'bt1', 'sp0', sh1']
                if len(keys) == 5:
                    to_modify.update({
                        ('val', 'episodic_val'): True,
                        ('loss', 'loss_type'): 'ce_loss',
                        ('network', 'model_name'): 'pspnet_contrast',
                        ('finetune', 'bottle_conv'): int(keys[0][1]),
                        ('finetune', 'bottle_lr_time'): int(keys[1][1])/10,  #  eg. 0.1 / 0.3
                        ('finetune', 'bottle_ft'): int(keys[1][1])!=0,   # if lr times is 00, not finetune
                        ('finetune', 'bottle_type'): int(keys[2][2]),  # 0 | 1 | 2 (orginal one)
                        ('train_split', ): int(keys[3][2]),          # 0 | 1 | 2 | 3
                        ('shot', ): int(keys[4][2]),
                        ('network', 'resume'): self.get('network', 'resume') + exp_id + '/max_performance.pth'
                    })
                elif len(keys) == 4:  # ['c3', '01', 'sp~', sh1]
                    to_modify.update({
                        ('val', 'episodic_val'): True,
                        ('loss', 'loss_type'): 'ce_loss',
                        ('network', 'model_name'): 'pspnet_contrast',
                        ('finetune', 'bottle_conv'): int(keys[0][1]),
                        ('finetune', 'bottle_lr_time'): int(keys[1][1])/10,  #  eg. 0.1 / 0.3
                        ('finetune', 'bottle_ft'): int(keys[1][1])!=0,   # if lr times is 00, not finetune
                        ('finetune', 'bottle_type'): 2,  # 0 | 1 | 2 (orginal one)
                        ('train_split', ): int(keys[2][2]),          # 0 | 1 | 2 | 3
                        ('shot', ): int(keys[3][2]),
                        ('network', 'resume'): './results/2023-02-12:ft_psp_r50_ce_pascal/' + exp_id + '/max_performance.pth'
                    })
        elif exp_name == 'ft_psp_r50_em':
            # c1_01_bt1_e5_sp0_sh1  conv 1x1, lr times 0.1, vacillate iter 50,  bottleneck type: 1, split 0, shot 1
            keys = exp_id.split('_')    # ['c1', '01', 'bt1', 'e5', 'sp0', sh1']
            # pdb.set_trace()
            to_modify.update({
                        ('val', 'episodic_val'): True,
                        ('loss', 'loss_type'): 'ce_loss',
                        ('network', 'model_name'): 'pspnet_contrast',
                        ('finetune', 'bottle_conv'): int(keys[0][1]),
                        ('finetune', 'bottle_lr_time'): int(keys[1][1])/10,  #  eg. 0.1 / 0.3
                        ('finetune', 'bottle_ft'): int(keys[1][1])!=0,   # if lr times is 00, not finetune
                        ('finetune', 'bottle_type'): int(keys[2][2]),  # 0 | 1 | 2 (orginal one)
                        ('finetune', 'vaci_iter'): int(keys[3][1])*10,  # vacillate iter
                        ('train_split', ): int(keys[4][2]),          # 0 | 1 | 2 | 3
                        ('shot', ): int(keys[5][2])
                    })
        if exp_id == 'test':
            self.update(to_modify)
            return to_modify

        self.update(to_modify)
        return to_modify

DEBUG_CONFIG = {
    ('train', 'test_interval'): 40,
    ('train', 'display_iter'): 10,
    ('train', 'max_epoch'): 3,
    ('val', 'val_num'): 40
}

if __name__ == '__main__':
    pass