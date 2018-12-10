#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from torch.utils.checkpoint import *
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from .utils import fill_up_weights, CpBatchNorm2d
BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckXElastic(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckXElastic, self).__init__()
        cardinality = BottleneckX.cardinality
        self.elastic = (stride == 1 and planes < 1024)
        if self.elastic:
            # self.ups = nn.ConvTranspose2d(
            #     inplanes, inplanes, 4, stride=2, padding=1,
            #     output_padding=0, groups=inplanes, bias=False)
            # fill_up_weights(self.ups)
            self.down = nn.AvgPool2d(2, stride=2)
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        bottle_planes = planes * cardinality // 32

        self.conv1_d = nn.Conv2d(inplanes, bottle_planes // 2,
                                 kernel_size=1, bias=False)
        self.bn1_d = BatchNorm(bottle_planes // 2)
        self.conv2_d = nn.Conv2d(bottle_planes // 2, bottle_planes // 2, kernel_size=3,
                                 stride=stride, padding=dilation, bias=False,
                                 dilation=dilation, groups=cardinality // 2)
        self.bn2_d = BatchNorm(bottle_planes // 2)
        self.conv3_d = nn.Conv2d(bottle_planes // 2, planes,
                                 kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(inplanes, bottle_planes // 2,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes // 2)
        self.conv2 = nn.Conv2d(bottle_planes // 2, bottle_planes // 2, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality // 2)
        self.bn2 = BatchNorm(bottle_planes // 2)
        self.conv3 = nn.Conv2d(bottle_planes // 2, planes,
                               kernel_size=1, bias=False)

        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.__flops__ = 0

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out_d = x
        if self.elastic:
            if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
                out_d = F.pad(out_d, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate')
            out_d = self.down(out_d)

        out_d = self.conv1_d(out_d)
        out_d = self.bn1_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv2_d(out_d)
        out_d = self.bn2_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv3_d(out_d)
        if self.elastic:
            out_d = self.ups(out_d)
            self.__flops__ += np.prod(out_d[0].shape) * 8
            if out_d.size(2) > x.size(2) or out_d.size(3) > x.size(3):
                out_d = out_d[:, :, :x.size(2), :x.size(3)]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out = out + out_d
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, seg=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, seg=seg)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, seg=seg)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride, ceil_mode=seg)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BottleneckX, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False, seg=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.seg = seg
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root, seg=seg)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, seg=seg)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, seg=seg)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, seg=seg)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride, ceil_mode=self.seg),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            if self.seg:
                x = checkpoint(getattr(self, 'level{}'.format(i)), x)
            else:
                x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return x


def dla60x(**kwargs):
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    return model


def dla102x(**kwargs):
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    return model


def dla60x_elastic(**kwargs):
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckXElastic, **kwargs)
    return model


def dla102x_elastic(**kwargs):
    BottleneckX.cardinality = 50
    model = DLA([1, 1, 3, 3, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckXElastic, residual_root=True, **kwargs)
    return model
