# -*- coding: utf-8 -*-
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


# from unet.unet_pyramid import UNet_Pyramid
# from TransUnet.transunet import TransUNet
# from SwinUnet.swin_unet_cTrans import SwinUnet

# from UCTransNet.UC_kaiming import UCTransNet
from UCTransNet.SE_en_de import UCTransNet
# from UCTransNet.UCTransNet import UCTransNet


# from unet.res_unet import ResUnet
# # from unet.SwinUnet(3Dunet import ResUNet
# from unet.MHConvNeXtUNetAttention import UNext
# from unet.swin_unet import SwinTransformerSys
from unet.unet_model import UNet
# from models.binary_deeplab import DeepLabV3

from utils.focal_loss import Focal_Loss
from utils.dice_loss import dice_loss
# from utils.Tloss import TLoss
from utils.dice_score import single_dice_score, dice_loss, multi_dice_score
from utils.feature_map import visualize_feature_map
import dataset
import numpy as np

dir_checkpoint = Path('./models/checkpoints/')
dice_ce_rate = 0.2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--n_classes', type=int, default=5)

    # parser.add_argument('--n_channels', type=int, default=8)
    # parser.add_argument('--n_classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', type=bool, default=True, help="if speed calculate")
    parser.add_argument('--save_checkpoint', type=bool, default=True, help="if save checkpoint")
    parser.add_argument('--batch_size', type=int, default=32)
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
    train_loader = DataLoader(dataset=dataset.genDataSet("train", n_classes), batch_size=batch_size, shuffle=False)
    print("batch_size{}".format(batch_size))
    print("learning_rate{}".format(learning_rate))
    val_loader = DataLoader(dataset=dataset.genDataSet("train", n_classes), batch_size=batch_size, shuffle=False)
    # 实验看板
    data_name = "a549肺癌细胞"
    if model.n_classes == 2:
        data_name = "鼻咽癌细胞"

    # 优化器 
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, eps= 1e-8, weight_decay=1e-8, momentum=0.9) #weight_decay 防止过拟合 #foreach=True
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-8,amsgrad=False)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)   # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    
   
   # goal: maximize Dice score
 
    
    
    
    best_loss = float('inf')
    # 训练开始
    # 5. Begin training
    train_losss, train_dices, test_losss, test_dices = [], [],[], []
    OAs, AAs, Kappas, acc0s, acc1s, acc2s, acc3s, acc4s, acc5s, acc6s = [], [], [], [], [], [], [], [], [], []
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        train_loss, train_dice, test_loss, test_dice = 0, 0, 0, 0
        for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            x, y = x.squeeze(1).to(device=device, dtype=torch.float32), y.squeeze(1).to(device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=amp):
                y_ = model(x)
                
                
                focal_loss = Focal_Loss(weight=0.25, gamma=2)
                
                
                # single_train_loss = (1 - dice_ce_rate) * criterion(y_, y) + dice_ce_rate * dice_loss(F.softmax(y_, dim=1).float(), F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)
                # single_train_loss = 0.2 * criterion(y_, y) + 0.8 * dice_loss(F.softmax(y_, dim=1).float(), F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)
                
                single_train_loss = (1 - dice_ce_rate) * criterion(y_, y) + dice_ce_rate * focal_loss.forward(F.softmax(y_, dim=1).float(), F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float())
                
                # single_train_loss = focal_loss.forward(F.softmax(y_, dim=1).float(),
                                                                     # F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float())
                    
                # T_loss = TLoss()
                # single_train_loss = T_loss.forward(F.softmax(y_, dim=1).float(),
                #                                                      F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float(),)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(single_train_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += single_train_loss.item()
                single_train_dice = single_dice_score(y_, y, n_classes=args.n_classes)
                train_dice += single_train_dice
        train_loss, train_dice = train_loss / len(train_loader), train_dice / len(train_loader)
        if epoch % 1 == 0:
            
            test_dice, test_loss,OA, AA, Kappa, acc0, acc1, acc2, acc3, acc4 = multi_dice_score(model, val_loader, device, dice_ce_rate,batch_size=batch_size, n_classes=n_classes)
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
            acc2s.append(acc2)
            acc3s.append(acc3)
            acc4s.append(acc4)
            print("Valid  %d epoch loss: %.4f dice:%.4f OA:%.4f AA:%.4f Kappa:%.4f 细胞背景：%.4f  细胞核膜：%.4f 细胞核仁：%.4f 细胞膜：%.4f 细胞质基质：%.4f" % (epoch, test_loss, test_dice, OA, AA, Kappa, acc0, acc1, acc2, acc3, acc4))
            
            if test_loss < best_loss:
                best_loss = test_loss
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                # if best_loss < 0.3:
                torch.save(model.state_dict(),
                            str(dir_checkpoint / 'Focal{}_{}class_{}_epoch{}_dice{:.6f}_loss{:.6f}.pth'.format(dice_ce_rate, model.n_classes, model.name, epoch, test_dice, test_loss)))
                logging.info(f'Checkpoint {epoch} saved!')
        model.eval()
    # 训练结束

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
    setup_seed(3401)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', )
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:{}'.format(args.cuda))
    # device = torch.device('cpu')
    # model = UNet(n_channels=args.n_channels, n_classes=args.n_classes)
    # model = ResUnet(n_channels=args.n_channels, n_classes=args.n_classes)

    # model = UNet_Pyramid(n_channels=args.n_channels, n_classes=args.n_classes)

    # model = threeunet(n_channels=1, n_classes=args.n_classes)
    # model = SwinUnet(n_channels=args.n_channels, n_classes=args.n_classes)
    model = UCTransNet(n_channels=args.n_channels, n_classes=args.n_classes)
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
    
    
    # Resume = True
    Resume = False
    if Resume:
        model_weights_path = "./models/_5_UC_en_de_epoch100_dice0.813114_loss0.307040.pth"
        model.load_state_dict(torch.load(model_weights_path))
        
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