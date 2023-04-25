# Final Model for Video Segmentation

import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from models.modules import trunc_normal_
from models.vit import Translator, vit_base_patch16

from models.hrnet.hrnet import HRNet_W48
from models.pspnet.pspnet import PSPNet


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, block_num, spatio_kernel):
        samplings = []
        samplings.extend([True for i in range(block_num)])
        super(Decoder, self).__init__()
        self.dec = []
        for i in range(block_num):
            self.dec.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[i]))
        self.dec = nn.Sequential(*self.dec)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, x, enc1=None):
        x = self.dec[0](x)
        for i in range(1, len(self.dec)):
            x = self.dec[i](x)
        x = self.readout(x)
        return x


class Segmenter(nn.Module):
    def __init__(self, args, img_size=224, seq_len=11, enc_dim=768, shrink_embed=32,
                trans_embed=192, num_heads=12, mlp_ratio=4, num_layers=9, dec_blocks=2, drop=0.0, device='cuda'):
        super(Segmenter, self).__init__()
        
        self.enc = vit_base_patch16(
            img_size=img_size,
            num_classes=10,  # randomly assign one, we don't use it
            drop_path_rate=0.1
        )

        # shrink the embedding dimension to save computation in ViT
        self.shrink = nn.Conv2d(enc_dim, shrink_embed, kernel_size=1, stride=1)
        
        # project the features that concat mask back to shrink embed
        self.proj_catmsk = nn.Conv2d(args.num_cls+shrink_embed, shrink_embed, kernel_size=1, stride=1)

        # nn.Conv2d(enc_dim, shrink_embed, kernel_size=5, stride=2)
        self.shrink_linear = nn.Linear(14*14*shrink_embed, trans_embed)

        self.translator = Translator(seq_len, trans_embed, num_heads, mlp_ratio*trans_embed, num_layers, drop, device)

        # expand
        self.expand_linear = nn.Linear(trans_embed, 14*14*shrink_embed)

        if args.seghead == 'hrnet48':
            # decoder for the first 11 frames' embedding
            self.f11_dec = Decoder(shrink_embed, 256, dec_blocks, spatio_kernel=5)
            # decoder for the second 11 frames' embedding
            self.s11_dec = Decoder(shrink_embed, 256, dec_blocks, spatio_kernel=5)

            self.seghead = HRNet_W48(args)  # input:  [B, 256, 56, 56]

        elif args.seghead == 'pspnet':
            # decoder for the first 11 frames' embedding
            self.f11_dec = Decoder(shrink_embed, 2048, dec_blocks, spatio_kernel=5)
            # decoder for the second 11 frames' embedding
            self.s11_dec = Decoder(shrink_embed, 2048, dec_blocks, spatio_kernel=5)
            
            self.seghead = PSPNet(args)   # input: [B, 2048, 56, 56]

        # initialize
        # [self.shrink, self.proj_catmsk, self.shrink_linear, self.translator, self.expand_linear, self.f11_dec, self.s11_dec, self.seghead]
        for modules in [self.proj_catmsk, self.f11_dec, self.s11_dec, self.seghead]:
            for m in modules.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def forward_encoder(self, x):
        # x.shape: B, T, C, H, W   B, 11, 3, 224, 224
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.enc(x) # (B T), 14*14, 768
        seq_len = int(x.shape[1] ** 0.5)

        x = rearrange(x, 'b (h w) d -> b d h w', h=seq_len) # (B T), 768, 14, 14

        # project to a lower dim
        x = self.shrink(x)   # (B T), shrink_embed, 14, 14
        
        return x


    def concate_mask(self, x, catmask):
        # x.shape: (B T), shrink_embed, 14, 14
        # catmask: B, 11, num_cls, 14, 14   11 because only has first 11 frames' mask
        catmask = rearrange(catmask, 'b t c h w -> (b t) c h w')
        x = torch.cat((x, catmask), dim=1) # (B T), shrink_embed+num_cls, 14, 14
        x = self.proj_catmsk(x)  # (B T), shrink_embed, 14, 14
        return x


    def forward_translator(self, x, T):
        _, d, h, _ = x.shape # (B T), shrink_embed, 14, 14

        x = rearrange(x, '(b t) d h w -> b t (d h w)', t=T) # B, T, 14*14*shrink_embed
        x = self.shrink_linear(x)  # B, T, trans_embed

        x = self.translator(x)  # B, T, trans_embed

        x = self.expand_linear(x) # B, T, 14*14*shrink_embed  
        x = rearrange(x, 'b t (d h w) -> (b t) d h w', d=d, h=h) # (B T), shrink_embed, 14, 14
        # x = self.expand(x)  # (B T), 128, 14, 14
        return x

    # first 11 frames' decoder
    def f11_forward_decoder(self, x):
        # predict the frames
        x = self.f11_dec(x)
        
        return x   # (B T), shrink_embed, 56, 56

    # sceond 11 frames' decoder
    def s11_forward_decoder(self, x, identity, skip=False):
        # predict the frames
        if skip:
            x = self.s11_dec(x + identity)
        else:
            x = self.s11_dec(x)
        
        return x   # (B T), shrink_embed, 56, 56


    def forward(self, x, catmask, label, skip=True, train=True):
        B, T, _, _, _ = x.shape
        # MAE encoder
        x = self.forward_encoder(x)

        x = self.concate_mask(x, catmask)
        identity = x   # frist 11 frames' embedding

        x = self.forward_translator(x, T)

        x = self.s11_forward_decoder(x, identity, skip=skip)     # (B 11), shrink_embed, 56, 56
        identity = self.f11_forward_decoder(identity)  # (B 11), shrink_embed, 56, 56

        x = torch.cat((identity, x), dim=0)   # (B 22), shrink_embed, 56, 56

        if not train:
            return x

        x = self.seghead(x)['seg']  # (B T), 49, 56, 56
        
        return self.forward_loss(x, label)


    def forward_loss(self, x, label):
        label = rearrange(label, 'b t c h w -> (b t) c h w')
        # x: (B T), 49, 56, 56
        x = F.interpolate(x, size=label.shape[-1], mode='bilinear', align_corners=True)
        # x: (B T), 49, 224, 224

        return F.cross_entropy(x, label), x
    

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.zero_()