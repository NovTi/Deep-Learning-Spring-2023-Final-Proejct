import os
import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer

from functools import partial
from einops import rearrange, repeat

from models.modules import trunc_normal_
from utils.util import interpolate_pos_embed


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        # TODO
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3)
        self.att_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # TODO
        # get the q, k, v
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d', h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2] # [b, h, n, d]

        # get the attention score
        att = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1]**0.5)
        if mask is not None:
            # att = att.masked_fill(mask == 0, -1e9)
            att += mask
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        out = torch.matmul(att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return self.proj_drop(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, mask, dropout):
        super().__init__()
        # TODO
        self.norm1 = nn.LayerNorm(embed_dim)
        self.MSA = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mask = mask

    def forward(self, x):
        # TODO
        x = x + self.MSA(self.norm1(x), mask=self.mask)
        
        x = x + self.MLP(self.norm2(x))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, embed_dim=768, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.out_channels = embed_dim
        self.head = nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 1:, :]  # remove cls token

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        """
        Parameters:
            d_model: dimension of input embeddings
            dropout (you don't need to use this, already in forward()): nn.Dropout
            max_len: maximum length of input
        Set variable value:
            pe: torch.tensor of size (max_len, d_model)
            
        """
        pe = torch.zeros(max_len, d_model)  # max_len, d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000.0 ** (-(torch.arange(0, d_model, 2).float() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Do not modify this.
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)   


class Translator(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, mlp_dim, num_layers, dropout, device, learn_pos_embed):
        super(Translator, self).__init__()
        # frist we try the learnable pos embedding
        self.learn_pos_embed = learn_pos_embed
        if learn_pos_embed:  # learnable pos embed
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        else: # not learnable pos embed
            self.pos_embedding = PositionalEncoding(embed_dim, dropout)
        mask = generate_square_subsequent_mask(seq_len).to(device)
        layers = []
        for i in range(num_layers):
            layers.append(TransformerBlock(embed_dim, num_heads, mlp_dim, mask, dropout))
        self.encode_blocks = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x):
        if self.learn_pos_embed:
            x = x + self.pos_embedding
        else:
            x = self.pos_embedding(x)

        return self.encode_blocks(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

