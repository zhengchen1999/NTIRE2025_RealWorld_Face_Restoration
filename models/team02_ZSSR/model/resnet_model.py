import torch
from torch import nn

# 采用ResNet替代原有的卷积层
class ZSSR_RES(nn.Module):
    def __init__(self, in_channels=3, channels=64, num_layer=4):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=channels,
            kernel_size=3, stride=1, padding=1,
        )

        resblock_list = []
        for i in range(num_layer):
            resblock_list.append(ResnetBlock(in_channels=channels))
        self.resblock_list = nn.Sequential(*resblock_list)

        self.out_conv = nn.Conv2d(
            in_channels=channels, out_channels=in_channels,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(self, x):
        f_in = self.in_conv(x)
        f_res = self.resblock_list(f_in) + f_in
        f_out = self.out_conv(f_res)
        return nn.functional.tanh(f_out) * 0.5 + 0.5





# ResnetBlock的子模块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1, padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block1 = Block(in_channels=self.in_channels, out_channels=self.out_channels)
        self.block2 = Block(in_channels=self.out_channels, out_channels=self.out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
