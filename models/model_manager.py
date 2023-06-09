# Video MAE for stage one reconstruction pretraining
from models.vmae.video_mae import pretrain_videomae_small_patch16_224
from models.vmae.video_mae import pretrain_videomae_base_patch16_224
from models.vmae.video_mae import pretrain_videomae_large_patch16_224
from models.vmae.video_mae import pretrain_videomae_huge_patch16_224

# MAE for stage one reconstruction pretraining
from models.mae.models_mae import mae_vit_base_patch16_dec512d8b
from models.mae.models_mae import mae_vit_large_patch16_dec512d8b
from models.mae.models_mae import mae_vit_huge_patch14_dec512d8b

# CViTVP Model for stage two video prediction pretraining
from models.cvitvp.cvitvp import CViT_VP

# Segmenter for video prediction segmentation
from models.segmenter.segmenter import Segmenter


MODEL = {
    'pretrain_videomae_small_patch16_224': pretrain_videomae_small_patch16_224,
    'pretrain_videomae_base_patch16_224': pretrain_videomae_base_patch16_224,
    'pretrain_videomae_large_patch16_224': pretrain_videomae_large_patch16_224,
    'pretrain_videomae_huge_patch16_224': pretrain_videomae_huge_patch16_224,
    'mae_vit_base_patch16': mae_vit_base_patch16_dec512d8b,
    'mae_vit_large_patch16': mae_vit_large_patch16_dec512d8b,
    'mae_vit_huge_patch16': mae_vit_huge_patch14_dec512d8b,
    'cvitvp': CViT_VP,
    'segmenter': Segmenter
}


class ModelManager(object):
    def __init__(self):
        pass
    
    def get_model(self, model_name):
        return MODEL[model_name]
