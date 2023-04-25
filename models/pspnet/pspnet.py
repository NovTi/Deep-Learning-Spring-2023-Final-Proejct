
import pdb
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

from models.module_helper import ModuleHelper
from models.pspnet.pspnet_modules import PSPModule
from models.pspnet.pspnet_modules import ProjectionHead


class PSPNet(nn.Module):
    def __init__(self, args):
        super(PSPNet, self).__init__()

        self.num_classes = args.num_cls
        self.bn_type = args.bn_type
        self.proj_dim = args.proj_dim

        in_channels = 2048

        # self.proj_head = ProjectionHead(
        #     dim_in=in_channels,
        #     proj_dim=self.proj_dim,
        #     bn_type=self.bn_type
        # )

        self.psp_head = PSPModule(  # bottleneck in this
            features=in_channels,
            out_features=in_channels//4,
            bn_type=self.bn_type
        )

        # take the bottleneck out for the finetuning / adjust bottleneck type
        # 0: doesn't have dropout2d / 1: has dropout2d
        if args.bottle_type == 0:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels+4*(in_channels//4), in_channels//4, 
                            kernel_size=args.bottle_conv, padding=1, dilation=1, bias=False),
                nn.ReLU())
        elif args.bottle_type == 1:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels+4*(in_channels//4), in_channels//4, \
                            kernel_size=args.bottle_conv, padding=1, dilation=1, bias=False),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            )
        else:  # original one
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels+4*(in_channels//4), in_channels//4, 
                            kernel_size=args.bottle_conv, padding=1, dilation=1, bias=False),
                ModuleHelper.BNReLU(in_channels//4, bn_type=self.bn_type),
                # nn.Dropout2d(0.1)
            )

        if args.classifier == 'simple':
            self.classifier = nn.Sequential(OrderedDict([('cls', nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True))]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False)),
                ('bn1', ModuleHelper.BatchNorm2d(bn_type=self.bn_type)(512)),
                ('cls', nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True)),
            ]))
            
        # self.proj_head
        for modules in [self.psp_head, self.classifier]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        # x = self.proj_head(x)

        # psp module
        x_psp = self.psp_head(x)

        # bottleneck
        x_bt = self.bottleneck(torch.cat(x_psp, 1))
        
        # seg module
        x_seg = self.classifier(x_bt)

        # return {'embed': embedding, 'seg': , 'h': x_psp}
        return {'x_bt': x_bt, 'seg': x_seg, 'x_psp': x_psp}

