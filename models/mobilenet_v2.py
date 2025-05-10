# coding: utf-8

from __future__ import division

"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks.
Copyright (c) Yang Lu, 2017; Modified By cleardusk
"""
import math
import torch
import torch.nn as nn

__all__ = ['MobileNetV2', 'mobilenetv2']

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, widen_factor=1.0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio * widen_factor))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # Depthwise convolution
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # Pointwise projection (linear, no ReLU)
            nn.Conv2d(hidden_dim, int(oup * widen_factor), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(oup * widen_factor)),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=62, input_channel=3, size=120, mode='small'):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
            input_channel: number of input channels (default 3 for RGB)
            size: input image size (e.g., 120 for 120x120)
            mode: model mode (e.g., 'small')
        """
        super(MobileNetV2, self).__init__()

        # MobileNetV2 configuration: (t, c, n, s)
        # t: expansion factor, c: output channels, n: number of repeats, s: stride
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Initial convolution
        self.conv1 = ConvBNReLU(input_channel, int(32 * widen_factor), stride=2)

        # Inverted residual blocks
        layers = []
        in_channels = int(32 * widen_factor)
        for t, c, n, s in self.cfgs:
            out_channels = int(c * widen_factor)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, stride, t, widen_factor))
                in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Final convolution
        self.conv2 = ConvBNReLU(in_channels, int(1280 * widen_factor), kernel_size=1)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1280 * widen_factor), num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.size = size
        self.mode = mode
        self.widen_factor = widen_factor

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mobilenetv2(**kwargs):
    """
    Construct MobileNetV2.
    """
    model = MobileNetV2(
        widen_factor=kwargs.get('widen_factor', 1.0),
        num_classes=kwargs.get('num_classes', 62),
        input_channel=kwargs.get('input_channel', 3),
        size=kwargs.get('size', 120),
        mode=kwargs.get('mode', 'small')
    )
    return model

# Keep existing MobileNet V1 functions for compatibility
def mobilenet_1(num_classes=62, input_channel=3):
    from .mobilenet_v1 import MobileNet  # Import MobileNet V1 from original file
    return MobileNet(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)

def mobilenet_05(num_classes=62, input_channel=3):
    from .mobilenet_v1 import MobileNet
    return MobileNet(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)

# Add MobileNetV2 variants
def mobilenetv2_1(num_classes=62, input_channel=3):
    return MobileNetV2(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)

def mobilenetv2_05(num_classes=62, input_channel=3):
    return MobileNetV2(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)