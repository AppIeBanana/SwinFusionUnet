import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from CTrans import ChannelTransformer  # 单纯测试网络时用的
import numpy as np
import matplotlib.pyplot as plt


def get_activation(activation_type):  # 用于指定激活函数的类型
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn,
                       activation_type)()  # 如果activation_type是torch.nn模块中的有效属性，函数将使用getattr()函数获取该属性对应的激活函数，并返回该激活函数的实例。getattr()函数可以根据属性名称动态获取对象的属性。
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    # 用于创建一个由多个卷积层组成的神经网络模块。
    #
    # 输入参数：
    # in_channels：输入通道数，表示输入数据的深度或通道数。
    # out_channels：输出通道数，表示输出数据的深度或通道数。
    # nb_Conv：卷积层的数量，表示要创建的卷积层的个数。
    # activation（可选）：激活函数的名称，默认为RelU.
    layers = []
    # 添加一个制定卷积层
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    # 在每次迭代中，它将一个包含有批归一化和指定激活函数的卷积层添加到 layers 列表中。这样做是为了根据 nb_Conv 参数创建额外的卷积层。
    for _ in range(nb_Conv - 1):  # 就是遍历的意思
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    # 最后，该函数返回一个从 layers 列表构建的 nn.Sequential 模块，该模块表示由所需数量的卷积层组成的神经网络模块。
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        # self.SwinTransformerBlock = SwinTransformerBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """
    CCA Block
    """

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)  # nb_Conv创建卷积层的数量

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class UCTransNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.name = "UCTransNet"
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.mtc = ChannelTransformer(vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=[16, 8, 4, 2])
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        fea_maps = [x1.detach().cpu(), x2.detach().cpu(), x3.detach().cpu(), x4.detach().cpu(), x5.detach().cpu()]

        x1, x2, x3, x4 = self.mtc(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        fea_maps.append(x.detach().cpu())
        x = self.up3(x, x3)
        fea_maps.append(x.detach().cpu())
        x = self.up2(x, x2)
        fea_maps.append(x.detach().cpu())
        x = self.up1(x, x1)
        fea_maps.append(x.detach().cpu())
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
            fea_maps.append(logits.detach().cpu())
        else:
            logits = self.outc(x)  # if nusing BCEWithLogitsLoss or class>1
            fea_maps.append(logits.detach().cpu())
        if self.vis:  # visualize the attention maps
            # return logits, att_weights, fea_maps
            # return logits, att_weights
            return logits
        else:
            # return logits, fea_maps
            return logits


if __name__ == '__main__':
    x = torch.randn(2, 8, 224, 224)
    model = UCTransNet(n_channels=8, n_classes=2)
    y = model(x)
    print(y.shape)
    # for feature in y2:
    #     feature = np.squeeze(feature, axis=0)
    #     plt.figure()
    #     plt.show()
    #     plt.imshow(feature[0])
    #     plt.show()
    #     print(feature.shape)
