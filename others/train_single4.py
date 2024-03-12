import argparse
import itertools
import logging
import os.path
from random import random
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.SwinTransformer import SwinTransformerSys
# from models.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from Utils.loss_ssim import ssim_loss
from Utils.loss_focal import Focal_Loss
from Utils.dice_score_U import single_dice_score, dice_loss, multi_dice_score
from dataset.dataset_U import MyDataSet
from pathlib import Path


# from skimage.metrics import peak_signal_noise_ratio as psnr

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--in_chans', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--amp', type=bool, default=True, help='if speed calculate')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='if save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def train_model(model, learning_rate, batch_size, loss_rate, epochs):
    torch.cuda.empty_cache()
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(dataset=MyDataSet('train'), batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    val_loader = DataLoader(dataset=MyDataSet('val'), batch_size=batch_size, shuffle=True, drop_last=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', )

    # 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, eps=args.eps, weight_decay=args.weight_decay,
                              momentum=0.9)  # weight_decay 防止过拟合 #foreach=True
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-8,amsgrad=False)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

    # amp只能在GPU环境下使用,因为amp是写在torch.cuda中的函数,
    # 而amp中包含了amp.GradScaler和amp.autocast函数
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    # Create an instance of the Focal_Loss class
    focal_loss = Focal_Loss(weight=0.25, gamma=2)
    # criterion = pytorch_ssim.SSIM(window_size=11)
    # 创建空列表用于保存损失
    best_loss = float('inf')
    best_dice = float('inf')
    writer = SummaryWriter(log_dir)
    # 训练开始
    train_losss, train_dices, test_losss, test_dices = [], [], [], []
    OAs, AAs, Kappas, acc0s, acc1s, acc2s, acc3s = [], [], [], [], [], [], []
    with open('../log.md', 'a') as f:
        f.write(
            f'----------------------------------------{datetime.date.today()}-----------------------------------\n'
            f'in_chans:{args.in_chans}, lr: {learning_rate}, eps: {args.eps}, weight_decay: {args.weight_decay}, batch_size:{batch_size},dice_ce_rate{loss_rate}\n')
    print('Start Train')
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        train_loss, train_dice, test_loss, test_dice = 0, 0, 0, 0
        for (x,label) in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            x,label = x.to(device, torch.float32), label.to(device, torch.long)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x)
                single_train_loss = (1 - loss_rate) * criterion(out, label) + loss_rate * dice_loss(
                    F.softmax(out, dim=1).float(),
                    F.one_hot(label, args.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True)
                # single_train_loss = (1 - loss_rate) * focal_loss.forward(
                #     F.softmax(out, dim=1).float(),
                #     F.one_hot(label, model.module.num_classes).permute(0, 3, 1, 2).float())
                # + loss_rate * ssim_loss(
                #     F.softmax(out, dim=1).float(),
                #     F.one_hot(label, model.module.num_classes).permute(0, 3, 1, 2).float(),
                #     window_size=21, size_average=True)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(single_train_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += single_train_loss.item()
                single_train_dice = single_dice_score(out, label, n_classes=args.num_classes)
                train_dice += single_train_dice
        train_loss, train_dice = train_loss / len(train_loader), train_dice / len(train_loader)
        if epoch % 1 == 0:
            test_dice, test_loss, OA, AA, Kappa, acc0, acc1, acc2, acc3 = multi_dice_score(model,val_loader,device,loss_rate,args.num_classes)
            scheduler.step(test_dice)
            logging.info('Validation Dice score: {}'.format(test_dice))

            print('Train %d epoch loss: %.4f dice:%.4f' % (epoch, train_loss, train_dice))
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
            # acc4s.append(acc4)
            print(
                'test  %d epoch loss: %.4f dice:%.4f OA:%.4f AA:%.4f Kappa:%.4f 未分类：%.4f  背景：%.4f 膜结构：%.4f 边缘：%.4f' % (
                    epoch, test_loss, test_dice, OA, AA, Kappa, acc0, acc1, acc2, acc3))
            # print(
            # 'test  %d epoch loss: %.4f dice:%.4f OA:%.4f AA:%.4f Kappa:%.4f 背景：%.4f  核膜：%.4f' % (
            #     epoch, test_loss, test_dice, OA, AA, Kappa, acc0, acc1))
            if test_loss < best_loss:
                best_loss = test_loss
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(dir_checkpoint / '_{}_SwinFUNet_BestLoss_class{}_lr{}_bt{}_rate{}_ep{}.pth'.format(
                               datetime.date.today(), model.module.num_classes, learning_rate, batch_size, loss_rate, epochs)))
                logging.info(f'Checkpoint {epoch} saved!')
                with open('../log.md', 'a') as f:
                    f.write(
                        f'best loss: {best_loss}, dice: {test_dice}, learning_rate:{learning_rate}, batchsize:{batch_size}, loss_rate:{loss_rate}\tepoch: {epoch}\n')
            if test_dice > best_dice:
                best_dice = test_dice
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(dir_checkpoint / '_{}_SwinFUNet_BestLoss_class{}_lr{}_bt{}_rate{}_ep{}.pth'.format(
                               datetime.date.today(), model.module.num_classes, learning_rate, batch_size, loss_rate, epochs)))
                logging.info(f'Checkpoint {epoch} saved!')
                with open('../log.md', 'a') as f:
                    f.write(
                        f'best loss: {best_loss}, dice: {test_dice}, learning_rate:{learning_rate}, batchsize:{batch_size}, loss_rate:{loss_rate}\tepoch: {epoch}\n')
        # tf-logs
        writer.add_scalar('train/loss', np.array(train_loss), epoch)
        writer.add_scalar('train/dice', np.array(train_dice.cpu()), epoch)
        writer.add_scalar('test/loss', np.array(test_loss.cpu()), epoch)
        writer.add_scalar('test/dice', np.array(test_dice.cpu()), epoch)
        writer.add_scalar('OA', np.array(OA), epoch)
        writer.add_scalar('AA', np.array(AA), epoch)
        writer.add_scalar('Kappa', np.array(Kappa), epoch)
        writer.add_scalar('acc0_背景', np.array(acc0), epoch)
        writer.add_scalar('acc1_核膜', np.array(acc1), epoch)
        writer.add_scalar('acc2_核仁', np.array(acc2), epoch)
        writer.add_scalar('acc3_细胞膜', np.array(acc3), epoch)
        # writer.add_scalar('acc4_基质', np.array(acc4), epoch)
        # predict_image = ''
        # label_image = ''
        # writer.add_image('Predict',predict_image,epoch)
        # writer.add_image('Truth',label_image,epoch)
        # 遍历模型的所有参数
        # for name, param in model.named_parameters():
        #     # 检查参数是否有梯度
        #     if param.requires_grad:
        #         # 获取参数的值
        #         values = param.data.cpu().numpy()
        #         # 添加权重直方图
        #         writer.add_histogram(name, values, bins='auto', global_step=epoch)

        model.eval()
    new_dir = '{}'.format(datetime.date.today())
    each_loss_path = os.path.join(loss_path, new_dir)
    if not os.path.exists(each_loss_path):
        os.mkdir(each_loss_path)
    np.save(os.path.join(each_loss_path, 'train_losss.npy'), np.array(train_losss))
    np.save(os.path.join(each_loss_path, 'train_dices.npy'), np.array(train_dices))
    np.save(os.path.join(each_loss_path, 'test_losss.npy'), np.array(test_losss))
    np.save(os.path.join(each_loss_path, 'test_dices.npy'), np.array(test_dices))
    np.save(os.path.join(each_loss_path, 'OAs.npy'), np.array(OAs))
    np.save(os.path.join(each_loss_path, 'AAs.npy'), np.array(AAs))
    np.save(os.path.join(each_loss_path, 'Kappas.npy'), np.array(Kappas))
    np.save(os.path.join(each_loss_path, 'acc0s.npy'), np.array(acc0s))
    np.save(os.path.join(each_loss_path, 'acc1s.npy'), np.array(acc1s))
    np.save(os.path.join(each_loss_path, 'acc2s.npy'), np.array(acc2s))
    np.save(os.path.join(each_loss_path, 'acc3s.npy'), np.array(acc3s))
    # np.save(os.path.join(each_loss_path, 'acc4s.npy'), np.array(acc4s))
    torch.cuda.empty_cache()
    writer.close()
    with open('../log.md', 'a') as f:
        f.write(f'-----------------------------------------------------------------------------------\n')


if __name__ == '__main__':
    args = get_args()
    setup_seed(3401)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', )
    loss_path = 'loss/SwinTransformerSys'
    save_path = '../image'
    dir_checkpoint = Path('../checkpoints/SwinFUnet')
    weight_path = 'checkpoints/SwinFUnet/_2023-12-22_SwinFUNet_best_loss_class5_loss0.8558_dice0.2558.pth'
    log_dir = '../tf-logs'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = UNet(n_channels=args.n_channels, num_classes=args.num_classes)
    # model = ResUnet(n_channels=args.n_channels, num_classes=args.num_classes)
    # model = SwinTransformerFusion(img_size=224, patch_size=4, in_chans_x=args.in_chans_x, in_chans_y=args.in_chans_y,
    #                               num_classes=args.num_classes,
    #                               embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
    #                               num_heads=[3, 6, 12, 24],
    #                               window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                               norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #                               use_checkpoint=False, final_upsample='expand_first').to(device)
    # model = threeunet(n_channels=1, num_classes=args.num_classes)
    # model = SwinUnet(n_channels=args.n_channels, num_classes=args.num_classes)
    # model = UCTransNet(n_channels=args.n_channels, num_classes=args.num_classes)
    # model = ResUNet(n_channels=1, num_classes=7, training=True)

    # 出现swin_unet has no attribute num_classes, n_channels 是因为swin_unet的class的init函数里没有定义self.n_channels=channels,self.classes=classes.
    # SwinTransformerSys
    model = SwinTransformerSys(img_size=512,
                               patch_size=4,
                               in_chans=args.in_chans,
                               num_classes=args.num_classes,
                               embed_dim=96,
                               depths=[2, 2, 2, 2],
                               depths_decoder=[1, 2, 2, 2],
                               num_heads=[3, 6, 12, 24],
                               window_size=16,
                               mlp_ratio=4.,
                               qkv_bias=True,
                               qk_scale=None,
                               drop_rate=0.,
                               attn_drop_rate=0.,
                               drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm,
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False,
                               final_upsample='expand_first')

    # model = UNext(num_classes=args.num_classes, n_channels=args.n_channels)

    # TransNet
    # model = TransUNet(img_dim=224,
    #                       n_channels=args.n_channels,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       num_classes=args.num_classes)

    # DeepLabV3
    # model = DeepLabV3(n_channels=args.n_channels, num_classes=args.num_classes, backbone='resnet50')

    # unet
    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.module.num_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')

    # swin_unet(error: swin_unet has no attribute in_chans)
    # logging.info(f'Network:\n'
    #              f'\t{model.in_chans} input channels\n'
    #              f'\t{model.module.num_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')

    # UNext
    # logging.info(f'Network:\n'
    #              f'\t{model.input_channels} input channels\n'
    #              f'\t{model.module.num_classes} output channels (classes)\n'
    #              # f'\t{model.name} Network\n'
    #              f'\t{device} device')
    # } Network\n')
    #    f'\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling')

    # if os.path.exists(weight_path):
    #     model.load_state_dict(torch.load(weight_path))
    #     print('weight load successful')
    # else:
    #     print('weight load defeat')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)    
    model.to(device)

    learning_rates = [1e-4, 2e-4]
    batch_sizes = [16]
    epochs = [50]
    loss_rates = [0.8, 0.9]
    # 获取所有参数组合的迭代器
    param_combinations = itertools.product(learning_rates, batch_sizes, epochs, loss_rates)
    # 循环遍历参数组合并训练模型
    for params in param_combinations:
        learning_rate, batch_size, epoch, loss_rate = params
        train_model(model=model, learning_rate=learning_rate, batch_size=batch_size, loss_rate=loss_rate, epochs=epoch)
