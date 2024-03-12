import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Dark_DownSample(nn.Module):
    def __init__(self, channel):
        super(Dark_DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(x)
        x = self.pool(x)
        return x


class Dark_UpSample(nn.Module):
    def __init__(self, channel):
        super(Dark_UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, low_level_features):
        x = self.upsample(x)
        x = self.layer(x)
        x = torch.cat([x, low_level_features], dim=0)
        return x


class Bright_DownSample(nn.Module):
    def __init__(self, channel):
        super(Bright_DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(x)
        x = self.pool(x)
        return x


class Bright_UpSample(nn.Module):
    def __init__(self, channel):
        super(Bright_UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample(x)
        return x


# 定义完整的Unet网络
class FUNet(nn.Module):
    def __init__(self, num_class):
        super(FUNet, self).__init__()
        # 暗场分支下采样
        self.dark_c1 = Conv_Block(16, 64)
        self.dark_down1 = Dark_DownSample(64)

        self.dark_c2 = Conv_Block(64, 128)
        self.dark_down2 = Dark_DownSample(128)

        self.dark_c3 = Conv_Block(128, 256)
        self.dark_down3 = Dark_DownSample(256)

        self.dark_c4 = Conv_Block(256, 512)
        self.dark_up1 = Dark_UpSample(512)

        self.dark_c5 = Conv_Block(512, 256)
        self.dark_up2 = Dark_UpSample(256)

        self.dark_c6 = Conv_Block(256, 128)
        self.dark_up3 = Dark_UpSample(128)

        self.dark_c7 = Conv_Block(128, 64)

        # 明场分支下采样
        self.bright_c1 = Conv_Block(16, 64)
        self.bright_down1 = Bright_DownSample(64)

        self.bright_c2 = Conv_Block(64, 128)
        self.bright_down2 = Bright_DownSample(128)

        self.bright_c3 = Conv_Block(128, 256)
        self.bright_down3 = Bright_DownSample(256)

        self.bright_c4 = Conv_Block(256, 512)
        self.bright_up1 = Bright_UpSample(512)

        self.bright_c5 = Conv_Block(512, 256)
        self.bright_up2 = Bright_UpSample(256)

        self.bright_c6 = Conv_Block(256, 128)
        self.bright_up3 = Bright_UpSample(128)

        self.bright_c7 = Conv_Block(128, 64)

        self.out = nn.Conv2d(64, num_class, 3, 1, 1)

    # Dark Bright Cat
    def forward(self, x, y):
        # 下采样
        D1 = self.dark_c1(x)  # 16*224*224 --> 64*224*224
        B1 = self.bright_c1(y)
        C1 = torch.cat((D1, B1), dim=0)
        # print(f'D1.shape:{D1.shape}'
        #       f'B1.shape:{B1.shape}'
        #       f'C1.shape:{C1.shape}')

        D2 = self.dark_c2(self.dark_down1(C1))  # 64*224*224 --> 64*112*112 --> 128*112*112
        B2 = self.bright_c2(self.bright_down1(B1))
        C2 = torch.cat((D2, B2), dim=0)
        # print(f'D2.shape:{D2.shape}'
        #       f'B2.shape:{B2.shape}'
        #       f'C2.shape:{C2.shape}')

        D3 = self.dark_c3(self.dark_down2(C2))  # 128*112*112 --> 128*56*56 --> 256*56*56
        B3 = self.bright_c3(self.bright_down2(B2))
        C3 = torch.cat((D3, B3), dim=0)
        # print(f'D3.shape:{D3.shape}'
        #       f'B3.shape:{B3.shape}'
        #       f'C3.shape:{C3.shape}')

        D4 = self.dark_c4(self.dark_down3(C3))  # 256*56*56 --> 256*28*28 --> 512*28*28
        B4 = self.bright_c4(self.bright_down3(B3))
        C4 = torch.cat((D4, B4), dim=0)
        # print(f'D4.shape:{D4.shape}'
        #       f'B4.shape:{B4.shape}'
        #       f'C4.shape:{C4.shape}')

        D5 = self.dark_up1(C4, C3)  # 512*28*28 --> 512*56*56 --> 256*56*56
        B5 = self.bright_c5(self.bright_up1(B4))
        C5 = torch.cat((D5, B5), dim=0)
        # print(f'D5.shape:{D5.shape}'
        #       f'B5.shape:{B5.shape}'
        #       f'C5.shape:{C5.shape}')

        # 上采样
        D6 = self.dark_up2(D5, C2)  # 256*56*56 --> 256*112*112 --> 128*112*112
        B6 = self.bright_c6(self.bright_up2(B5))
        C6 = torch.cat((D6, B6), dim=0)
        # print(f'D6.shape:{D6.shape}'
        #       f'B6.shape:{B6.shape}'
        #       f'C6.shape:{C6.shape}')

        D7 = self.dark_up3(D6, C1)  # 128*112*112 --> 128*224*224 --> 64*224*224
        B7 = self.bright_c7(self.bright_up3(B6))
        C7 = torch.cat((D7, B7), dim=0)
        # print(f'D7.shape:{D7.shape}'
        #       f'B7.shape:{B7.shape}'
        #       f'C7.shape:{C7.shape}')

        finnal_C = torch.cat((C7, D1), dim=0)
        out = self.out(finnal_C)  # 64*224*224 --> num_class*
        # print(f'out.shape:{out.shape}')
        return out


if __name__ == '__main__':
    x = torch.randn(4, 16, 224, 224)
    y = torch.randn(4, 16, 224, 224)
    net = FUNet(6)
    print(net(x, y).shape)
