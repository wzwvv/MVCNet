# --------------------------------------------------------
# IFNet [J. Wang, L. Yao, and Y. Wang, “IFNet: An interactive frequency con- volutional neural network for enhancing
# motor imagery decoding from EEG,” IEEE Trans. on Neural Systems and Rehabilitation Engineering, vol. 31, pp.
# 1900–1911, 2023.]
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(self, data_name, in_planes, out_planes = 64, kernel_size = 63, patch_size = 125, radix = 2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.data_name = data_name

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()

        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)
        if self.data_name != 'MI1-7':
            out = out[:, :, :-1]
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class IFNet(nn.Module):
    def __init__(self, data_name, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes):
        r'''Interactive Frequency Convolutional Neural Network

        :param in_planes: Number of input EEG channels
        :param out_planes: Number of output feature dimensions
        :param kernel_size: Temporal convolution kernel size
        :param radix:   Number of input frequency bands
        :param patch_size: Temporal pooling size
        :param time_points: Input window length
        :param num_classes: Number of classes
        '''
        nn.Module.__init__(self)
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.data_name = data_name
        self.stem = Stem(self.data_name, self.in_planes, self.out_planes, kernel_size, patch_size=patch_size, radix=radix)

        self.fc = nn.Sequential(
            nn.Linear(self.out_planes * (time_points // patch_size) , num_classes),
        )
        #print(f'fc layer feature dims:{self.fc[-1].weight.shape}')
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, C, T = x.shape
        out = self.stem(x)
        out = out.flatten(1)
        # out = self.fc(out.flatten(1))
        return out