# MAE Model for stage one reconstruction pretraining
from models.models_mae import mae_vit_base_patch16_dec512d8b
from models.models_mae import mae_vit_large_patch16_dec512d8b
from models.models_mae import mae_vit_huge_patch14_dec512d8b

# CViTVP Model for stage two video prediction pretraining
from models.cvitvp import CViT_VP

MODEL = {
    'mae_vit_base_patch16': mae_vit_base_patch16_dec512d8b,
    'mae_vit_base_patch16': mae_vit_large_patch16_dec512d8b,
    'mae_vit_base_patch16': mae_vit_huge_patch14_dec512d8b,
    'cvitvp': CViT_VP
}


class ModelManager(object):
    def __init__(self):
        pass
    
    def get_model(self, model_name):
        return MODEL[model_name]
