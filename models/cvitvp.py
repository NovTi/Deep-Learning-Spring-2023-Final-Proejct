import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from models.modules import trunc_normal_
from models.vit import Translator, vit_base_patch16


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


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, block_num, spatio_kernel):
        samplings = [False]
        samplings.extend([True for i in range(block_num-1)])
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC( C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, block_num, spatio_kernel):
        samplings = [False]
        samplings.extend([True for i in range(block_num-1)])
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


# convolution vit video prediction
class CViT_VP(nn.Module):
    def __init__(self, img_size=224, seq_len=11, enc_dim=768, shrink_embed=32,
                trans_embed=192, num_heads=12, mlp_ratio=4, num_layers=9, dec_blocks=3, drop=0.0, device='cuda'):
        super(CViT_VP, self).__init__()
        """
        input shape: B, T, C, H, W
            B: batch size  T: number of framess
        """
        
        self.enc = vit_base_patch16(
            img_size=img_size,
            num_classes=10,  # randomly assign one, we don't use it
            drop_path_rate=0.1
        )

        # shrink the embedding dimension to save computation in ViT
        self.shrink = nn.Conv2d(enc_dim, shrink_embed, kernel_size=1, stride=1)
        # nn.Conv2d(enc_dim, shrink_embed, kernel_size=5, stride=2)
        self.shrink_linear = nn.Linear(14*14*32, trans_embed)

        self.translator = Translator(seq_len, trans_embed, num_heads, mlp_ratio*trans_embed, num_layers, drop, device)

        # expand
        self.expand_linear = nn.Linear(trans_embed, 14*14*32)
        # self.expand = nn.Conv2d(shrink_embed, 64, kernel_size=1, stride=1)

        self.dec = Decoder(32, 3, dec_blocks, spatio_kernel=3)

        self.criterion = nn.MSELoss()

    def forward_encoder(self, x):
        # x.shape: B, T, C, H, W   B, 11, 3, 224, 224
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.enc(x) # (B T), 14*14, 768
        seq_len = int(x.shape[1] ** 0.5)

        x = rearrange(x, 'b (h w) d -> b d h w', h=seq_len) # (B T), 768, 14, 14

        # project to a lower dim
        x = self.shrink(x)   # (B T), 32, 14, 14
        
        return x

    def forward_translator(self, x, T):
        _, d, h, _ = x.shape

        x = rearrange(x, '(b t) d h w -> b t (d h w)', t=T) # B, T, 14*14*32
        x = self.shrink_linear(x)  # B, T, trans_embed (192)

        x = self.translator(x)  # B, T, trans_embed (192)

        x = self.expand_linear(x) # B, T, 14*14*32  
        x = rearrange(x, 'b t (d h w) -> (b t) d h w', d=d, h=h) # (B T), 32, 14, 14
        # x = self.expand(x)  # (B T), 128, 14, 14
        return x
    
    def forward_decoder(self, x, B, identity, skip=False):
        # predict the frames
        if skip:
            x = self.dec(x + identity)
        else:
            x = self.dec(x)
        
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B) # x.reshape(B, 11, 3, 224, 224)

        return x

    def forward(self, x, y, skip=False):
        B, T, C, H, W = x.shape
        # MAE encoder
        x = self.forward_encoder(x)
        identity = x
        x = self.forward_translator(x, T)
        x = self.forward_decoder(x, B, identity, skip=skip)
        loss = self.forward_loss(x, y, B)

        return loss

    def forward_loss(self, x, label, B, norm_pix_loss=False):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        pred = F.interpolate(x, size=label.shape[-1], mode='bilinear', align_corners=True)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=B)

        return self.criterion(pred, label)
        


# if __name__ == '__main__':
#     # note: trans_embed must be divisible by num_heads
#     test = CViT_VP(224, seq_len=11, enc_dim=768, shrink_embed=32, trans_embed=192,
#                     num_heads=12, mlp_ratio=2, num_layers=9, dec_blocks=4, drop=0.0)
#     a = torch.rand(4, 11, 3, 224, 224)
#     test(a)
