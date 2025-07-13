import torch
from torch import nn


class ZSSR_Net(nn.Module):
    '''
    参数设定：
    in_channels: 输入图像的通道数
    channels：   神经网络的隐层的通道数量
    num_layer:   CNN的层数
    '''
    def __init__(self, in_channels=3, channels=64, num_layer=8):
        super().__init__()
        self.in_channels = in_channels
        model_list = []
        for i in range(num_layer):
            in_channels = self.in_channels if i == 0 else channels
            out_channels = self.in_channels if i == num_layer - 1 else channels
            conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
            model_list.append(conv_block)
        self.model_list = nn.Sequential(*model_list)
        self.range_scale_layer = Range_Scale_Layer()

    def forward(self, x):
        feature = self.model_list(x) + x
        return self.range_scale_layer(feature)


class ConvBlock(nn.Module):
    '''
    一个卷积模块的基本结构：
    卷积层 -> 组归一化层 -> SiLU激活函数
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        GroupNorm = nn.GroupNorm(num_groups=8, num_channels=out_channels) if out_channels != 3 else nn.Identity()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1,
                padding_mode='reflect'
            ),
            GroupNorm,
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class Range_Scale_Layer(nn.Module):
    '''
    将输出的图像数值范围缩放到0-1的范围
    首先使用tanh函数将数值范围缩放到（-1， 1）
    然后进一步将范围调整到（0， 1）
    '''
    def __init__(self):
        super().__init__()
        self.scale_layer = nn.Tanh()

    def forward(self, x):
        return self.scale_layer(x) * 0.5 + 0.5


from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZSSR_Net().to(device)
    input = torch.randn(1, 3, 256, 256).to(device)

    out = model(input)
    print(out.shape)

    sw = SummaryWriter('./logs')
    sw.add_graph(model, input)
    sw.close()
