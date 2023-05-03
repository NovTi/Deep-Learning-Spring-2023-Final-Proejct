import torch
import torch.nn as nn
import torch.nn.functional as F


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        output = F.softmax(h, dim =1)
        return h


def get_Unet(n_classes = 49, load_weights=False):
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False, scale=1)
    net.outc = OutConv(64, n_classes)
    if load_weights:
        net.load_state_dict(torch.load("no_pretrain.pt"))
    return net