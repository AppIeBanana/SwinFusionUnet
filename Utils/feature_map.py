
import matplotlib.pyplot as plt
import numpy as np
from unet.unet_model import UNet
from unet.MHConvNeXtUNetAttention import UNext

import torch
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(feature_map):
    # 旨在输出某一层的所有通道？
    # feature_map一共两个通道，

    # squeeze之前shape为(1, 2, 224, 224) 【此时main函数里只有一张图片】，故batch_size为1。
    feature_map = np.squeeze(feature_map, axis=0)
    # squeeze之后shape为(2, 224, 224) , 两通道, 两通道(feature_map[0], feature_map[1])的shape均为(224, 224)。

    # 四行plt是师兄add的
    # 旨在直接输出整张特征图
    # 使用plt.figure()函数创建一个新的Figure对象, 该对象将作为画布
    plt.figure()
    # plt.imshow(feature_map[0, :, 0])
    plt.show()
    plt.imshow(feature_map[0])

    plt.show()
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    # IndexError: tuple index out of range
    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):

        # 问题出在这一行上（是输出第0个通道的...)
        # feature_map_split = feature_map[:, :, i]
        feature_map_split = feature_map[i ,:, :]
        # feature_map_split.shape: (224, 224)

        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)

        # 新add的一行，为了应对RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        # feature_map_split = feature_map_split.detach().numpy()


        plt.imshow(feature_map_split)
        plt.axis('off')
        plt.title('feature_map_{}'.format(i))

    plt.savefig('../result/feature_map_UNext_7.png')
    plt.show()


def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):#model的第一个Sequential()是有多层，所以遍历
            x = layer(x)#torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


if __name__ == "__main__":
    print("特征图可视化")

    # 二分类
    # net = UNet(n_channels=8, n_classes=2)
    # net = UNext(n_channels=8, n_classes=2)
    # model_save_path = "../models/checkpoints/_2_unet_epoch50_dice0.9410497546195984.pth"
    # model_save_path = "../models/checkpoints/_2_MHConvNeXtUNetAttention_epoch11_dice0.8100078105926514.pth"
    # net.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    # img_path = "../dataset/data.npy"
    # # 一共150张图片，所以index为0-149。
    # x = torch.from_numpy(np.load(img_path)[100]).to(torch.float32)   # torch.Size([8, 224, 224])
    # x = torch.unsqueeze(x, dim=0)   # torch.Size([1, 8, 224, 224])
    # y = net(x)
    # y = y.cpu().detach().numpy()    # torch.Size([1, 2, 224, 224])
    # visualize_feature_map(y)


    # 七分类
    # net = UNet(n_channels=10, n_classes=7)
    net = UNext(n_channels=10, n_classes=7)
    # model_save_path = "../models/checkpoints/_7_unet_epoch20_dice0.6415917873382568.pth"
    model_save_path = "../models/checkpoints/_7_MHConvNeXtUNetAttention_epoch43_dice0.8543124198913574.pth"
    net.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    img_path = "../dataset/data_sel_gold_400img_300.npy"
    # 一共有四百张图片，所以index为0-399
    x = torch.from_numpy(np.load(img_path)[350]).to(torch.float32)   # torch.Size([1, 10, 224, 224])
    # 新add的一行, 55难得解决的bug. 参考的https://blog.csdn.net/chengxy1998/article/details/117383054
    # x = torch.unsqueeze(x, dim=0)  # torch.Size([1, 1, 10, 224, 224])
    y = net(x)
    y = y.cpu().detach().numpy()
    visualize_feature_map(y)


    # x = x.cpu().detach().numpy()


    # print(net(x).shape)
    # print(net)