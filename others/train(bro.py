import torch
import logging
import argparse
import wandb
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from unet.unet_pyramid import UNet_Pyramid
from threeunet.UNet import threeunet
# from TransUnet.transunet import TransUNet
# # from SwinUnet.vision_transformer import SwinUnet
# from SwinUnet.swin_unet_cTrans import SwinUnet
# from UCTransNet.UCTransNet import UCTransNet
# from unet.res_unet import ResUnet
# # from unet.SwinUnet(3Dunet import ResUNet
# from unet.MHConvNeXtUNetAttention import UNext
# from unet.swin_unet import SwinTransformerSys
# from unet.unet_model import UNet
# from models.binary_deeplab import DeepLabV3
from utils.dice_loss import dice_loss
from utils.dice_score import single_dice_score, dice_loss, multi_dice_score
from utils.feature_map import visualize_feature_map
import dataset
import numpy as np

dir_checkpoint = Path('./models/checkpoints/')
dice_ce_rate = 0.8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--cuda', type=int, default=0)
    # parser.add_argument('--n_channels', type=int, default=10)
    # parser.add_argument('--n_classes', type=int, default=7)

    parser.add_argument('--n_channels', type=int, default=8)
    parser.add_argument('--n_classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', type=bool, default=True, help="if speed calculate")
    parser.add_argument('--save_checkpoint', type=bool, default=True, help="if save checkpoint6")
    parser.add_argument('--batch_size', type=int, default=2)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def train_model(model, epochs, batch_size, learning_rate, device, amp, n_classes):
    torch.cuda.empty_cache()
    # 数据集
    train_loader = DataLoader(dataset=dataset.genDataSet("train", n_classes), batch_size=batch_size, shuffle=True)

    # 格式化输出函数format()
    #  相对基本格式化输出采用‘%’的方法，format()功能更强大，该函数把字符串当成一个模板，
    #  通过传入的参数进行格式化，并且使用大括号‘{}’作为特殊字符代替‘%’
    print("learning_rate: {}".format(learning_rate))
    val_loader = DataLoader(dataset=dataset.genDataSet("val", n_classes), batch_size=batch_size, shuffle=True)
    # 实验看板
    # data_name = "a549肺癌细胞"
    if model.n_classes == 2:
        data_name = "鼻咽癌细胞"
        print(data_name)

    # 优化器
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-6,
    #                           momentum=0.9)  # weight_decay 防止过拟合 #foreach=True
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-8,amsgrad=False)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score


    # amp只能在GPU环境下使用,因为amp是写在torch.cuda中的函数,
    # 而amp中包含了amp.GradScaler和amp.autocast函数
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    # goal: maximize Dice score

    best_loss = float('inf')
    # 训练开始
    # 5. Begin training
    train_losss, train_dices, test_losss, test_dices = [], [], [], []
    OAs, AAs, Kappas, acc0s, acc1s, acc2s, acc3s, acc4s, acc5s, acc6s = [], [], [], [], [], [], [], [], [], []
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        train_loss, train_dice, test_loss, test_dice = 0, 0, 0, 0
        for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            # x:torch.Size([2, 8, 224, 224]), y:torch.Size([2, 224, 224])
            x, y = x.squeeze(1).to(device=device, dtype=torch.float32), y.squeeze(1).to(device=device, dtype=torch.long)
            # x = x.unsqueeze(axis=2)
            # with torch.cuda.amp.autocast(enabled=amp):

            if model.name == 'threeunet':
                x = x.unsqueeze(1)

            # y_:torch.Size([2, 2, 224, 224])
            y_ = model(x)

            # single_train_loss = (1 - dice_ce_rate) * criterion(y_, y) + dice_ce_rate * dice_loss(F.softmax(y_, dim=1).float(), F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)

            # single_train_loss = 1 * criterion(y_, y) + 5 * dice_loss(F.softmax(y_, dim=1).float(),
            #                                                          F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),
            #                                                          multiclass=True)

            single_train_loss = dice_loss(F.softmax(y_, dim=1).float(),
                                          F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),
                                          multiclass=True)


            optimizer.zero_grad(set_to_none=True)
            # grad_scaler.scale(single_train_loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()

            train_loss += single_train_loss.item()
            single_train_dice = single_dice_score(y_, y, n_classes=args.n_classes)
            train_dice += single_train_dice
        train_loss, train_dice = train_loss / len(train_loader), train_dice / len(train_loader)
        if epoch % 1 == 0:

            test_dice, test_loss, OA, AA, Kappa, acc0, acc1 = multi_dice_score(model, val_loader, device, dice_ce_rate)
            # test_dice, test_loss = multi_dice_score(model, val_loader, device, dice_ce_rate)
            scheduler.step(test_dice)
            logging.info('Validation Dice score: {}'.format(test_dice))

            print("Train %d epoch loss: %.4f dice:%.4f" % (epoch, train_loss, train_dice))
            train_losss.append(train_loss)
            train_dices.append(train_dice.cpu())
            test_losss.append(test_loss.cpu())
            test_dices.append(test_dice.cpu())
            OAs.append(OA)
            AAs.append(AA)
            Kappas.append(Kappa)
            acc0s.append(acc0)
            acc1s.append(acc1)
            print("test  %d epoch loss: %.4f dice:%.4f OA:%.4f AA:%.4f Kappa:%.4f 背景：%.4f  膜：%.4f" % (
            epoch, test_loss, test_dice, OA, AA, Kappa, acc0, acc1))
            # print("test  %d epoch loss: %.4f dice:%.4f" % (
            # epoch, test_loss, test_dice))


            if test_loss < best_loss:
                best_loss = test_loss
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(dir_checkpoint / '_{}_SwinUNet_epoch{}_dice{}.pth'.format(model.n_classes, epoch, test_dice)))
                logging.info(f'Checkpoint {epoch} saved!')

        # 验证和测试做model.eval()时,框架会自动把BN和DropOut固定住,不会取平均,而是用训练好的值
        # 在eval的时候要用”model.eval()”, 用来告诉网络现在要进入测试模式了
        model.eval()
    # 训练结束
    # epoch = range(1, epochs + 1)
    # plt.plot(epoch, train_losss)
    # plt.ylabel('train_loss')
    # plt.xlabel('epoch')
    # plt.show()

    import numpy as np
    np.save("train_losss.npy", np.array(train_losss))
    np.save("train_dices.npy", np.array(train_dices))
    np.save("test_losss.npy", np.array(test_losss))
    np.save("test_dices.npy", np.array(test_dices))
    np.save("OAs.npy", np.array(OAs))
    np.save("AAs.npy", np.array(AAs))
    np.save("Kappas.npy", np.array(Kappas))
    np.save("acc0s.npy", np.array(acc0s))
    np.save("acc1s.npy", np.array(acc1s))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    # setup_seed(3401)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', )
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:{}'.format(args.cuda))
    # device = torch.device('cpu')
    # model = UNet(n_channels=args.n_channels, n_classes=args.n_classes)
    # model = ResUnet(n_channels=args.n_channels, n_classes=args.n_classes)

    model = UNet_Pyramid(n_channels=args.n_channels, n_classes=args.n_classes)

    # model = threeunet(n_channels=1, n_classes=args.n_classes)
    # model = SwinUnet(n_channels=args.n_channels, n_classes=args.n_classes)
    # model = UCTransNet(n_channels=args.n_channels, n_classes=args.n_classes)
    # model = ResUNet(n_channels=1, n_classes=7, training=True)
    # 出现swin_unet has no attribute n_classes, n_channels 是因为swin_unet的class的init函数里没有定义self.n_channels=channels,self.classes=classes.
    # model = SwinTransformerSys(img_size=224,
    #                            patch_size=4,
    #                            n_channels=10,
    #                            n_classes=7,
    #                            embed_dim=96,
    #                            depths=[2, 2, 2, 2],
    #                            num_heads=[3, 6, 12, 24],
    #                            window_size=7,
    #                            mlp_ratio=4.,
    #                            qkv_bias=True,
    #                            qk_scale=None,
    #                            drop_rate=0.1,
    #                            drop_path_rate=0.1,
    #                            ape=False,
    #                            patch_norm=True,
    #                            use_checkpoint=False)
    # model = UNext(n_classes=args.n_classes, n_channels=args.n_channels)
    # model = TransUNet(img_dim=224,
    #                       n_channels=args.n_channels,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       n_classes=args.n_classes)
    # model = DeepLabV3(n_channels=args.n_channels, n_classes=args.n_classes, backbone='resnet50')
    model.to(device)


    # model.cuda()

    # unet
    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')

    # swin_unet(error: swin_unet has no attribute in_chans)
    # logging.info(f'Network:\n'
    #              f'\t{model.in_chans} input channels\n'
    #              f'\t{model.num_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')

    # UNext
    # logging.info(f'Network:\n'
    #              f'\t{model.input_channels} input channels\n'
    #              f'\t{model.num_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')
    # } Network\n')
    #    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    train_model(model=model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device,
                amp=args.amp, n_classes=args.n_classes)
