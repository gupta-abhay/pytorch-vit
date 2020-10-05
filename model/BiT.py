import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return StdConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return StdConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, in_planes, out_planes=None, mid_planes=None, stride=1):
        super(PreActBottleneck, self).__init__()
        out_planes = out_planes or in_planes
        mid_planes = mid_planes or out_planes // 4

        self.gn1 = nn.GroupNorm(32, in_planes)
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        self.conv2 = conv3x3(mid_planes, mid_planes, stride)
        self.gn3 = nn.GroupNorm(32, mid_planes)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(in_planes, out_planes, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual


class ResNetV2Model(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843):
        super(ResNetV2Model, self).__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=64*wf, out_planes=256*wf, mid_planes=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=256*wf, out_planes=256*wf, mid_planes=64*wf)) for i in range(2, block_units[0] + 1)],
        ))
        self.conv3 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=256*wf, out_planes=512*wf, mid_planes=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=512*wf, out_planes=512*wf, mid_planes=128*wf)) for i in range(2, block_units[1] + 1)],
        ))
        self.conv4 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=512*wf, out_planes=1024*wf, mid_planes=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=1024*wf, out_planes=1024*wf, mid_planes=256*wf)) for i in range(2, block_units[2] + 1)],
        ))
        self.conv5 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=1024*wf, out_planes=2048*wf, mid_planes=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=2048*wf, out_planes=2048*wf, mid_planes=512*wf)) for i in range(2, block_units[3] + 1)],
        ))
        # pylint: enable=line-too-long

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))

    def forward(self, x, include_conv5=False, include_top=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if include_conv5:
            x = self.conv5(x)
        if include_top:
            x = self.head(x)

        if include_top and include_conv5:
            assert x.shape[-2:] == (1,1,)
            return x[..., 0, 0]

        return x
