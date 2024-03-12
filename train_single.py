import logging
import os.path
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.SwinFUnet import SwinTransformerFusion
from Utils.loss_ssim import ssim_loss
from Utils.loss_focal import Focal_Loss
from Utils.dice_score_single import single_dice_score, dice_loss, multi_dice_score
from dataset.dataset_single import MyDataSet
from pathlib import Path


def train_single(args, model, device):
    torch.cuda.empty_cache()
    train_loader = DataLoader(dataset=MyDataSet('train', args.img_size, args.num_classes), batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              drop_last=True)
    val_loader = DataLoader(dataset=MyDataSet('val', args.img_size, args.num_classes), batch_size=args.batch_size,
                            shuffle=True, drop_last=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', )

    # 优化器
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, eps=args.eps, weight_decay=args.weight_decay,
    #                           momentum=0.9)  # weight_decay 防止过拟合 #foreach=True
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=args.eps,
                           weight_decay=args.weight_decay,
                           amsgrad=False)
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

    writer = SummaryWriter(args.log_dir)
    # 训练开始
    train_losss, train_dices, test_losss, test_dices = [], [], [], []
    OAs, AAs, Kappas, acc0s, acc1s, acc2s, acc3s, acc4s = [], [], [], [], [], [], [], []
    with open('./log.md', 'a') as f:
        f.write(
            f'----------------------------------------{datetime.date.today()}-----------------------------------\n'
            f'n_chans_x:{args.in_chans_x}, lr: {args.learning_rate}, eps: {args.eps}, weight_decay: {args.weight_decay}, batch_size:{args.batch_size},dice_ce_rate{args.dice_ce_rate}\n')
    print('Start Train')
    for epoch in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        train_loss, train_dice, test_loss, test_dice = 0, 0, 0, 0
        for (x, label) in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            x, label = x.to(device, torch.float32), label.to(device, torch.long)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x)
                single_train_loss = (1 - args.dice_ce_rate) * criterion(out, label) + args.dice_ce_rate * dice_loss(
                    F.softmax(out, dim=1).float(),
                    F.one_hot(label, args.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True)
                # single_train_loss = (1 - args.dice_ce_rate) * criterion(out, label) + args.dice_ce_rate * generalized_dice_loss(
                #     F.softmax(out, dim=1).float(),
                #     F.one_hot(label, args.num_classes).permute(0, 3, 1, 2).float())
                # single_train_loss = (1 - args.dice_ce_rate) * focal_loss.forward(
                #     F.softmax(out, dim=1).float(),
                #     F.one_hot(label, args.num_classes).permute(0, 3, 1, 2).float())
                # + args.dice_ce_rate * ssim_loss(
                #     F.softmax(out, dim=1).float(),
                #     F.one_hot(label, args.num_classes).permute(0, 3, 1, 2).float(),
                #     window_size=21, size_average=True)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(single_train_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += single_train_loss.item()
                single_train_dice = single_dice_score(out, label, num_classes=args.num_classes)
                train_dice += single_train_dice
        train_loss, train_dice = train_loss / len(train_loader), train_dice / len(train_loader)
        if epoch % 1 == 0:
            test_dice, test_loss, OA, AA, Kappa, acc0, acc1, acc2, acc3, acc4 = multi_dice_score(model,
                                                                                                 val_loader,
                                                                                                 device,
                                                                                                 args.dice_ce_rate,
                                                                                                 num_classes=args.num_classes)
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
            acc4s.append(acc4)
            print(
                'test  %d epoch loss: %.4f dice:%.4f OA:%.4f AA:%.4f Kappa:%.4f 核仁：%.4f  核膜：%.4f 细胞质：%.4f 细胞边缘：%.4f 背景: %.4f' % (
                    epoch, test_loss, test_dice, OA, AA, Kappa, acc0, acc1, acc2, acc3, acc4))

            if test_loss < best_loss:
                best_loss = test_loss
                Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(args.checkpoint_dir / '_{}_SwinFUNet_BestLoss_img{}_class{}_lr{}_bt{}_rate{}_ep{}.pth'.format(
                               datetime.date.today(), args.img_size, args.num_classes, args.learning_rate,
                               args.batch_size,
                               args.dice_ce_rate,
                               args.epochs)))
                logging.info(f'Checkpoint {epoch} saved!')
                with open('./log.md', 'a') as f:
                    f.write(
                        f'best loss: {best_loss}, dice: {test_dice}, learning_rate:{args.learning_rate}, batchsize:{args.batch_size}, dice_ce_rate:{args.dice_ce_rate}\tepoch: {epoch}\n')

            if test_dice > best_dice:
                best_dice = test_dice
                Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(args.checkpoint_dir / '_{}_SwinFUNet_BestLoss_img{}_class{}_lr{}_bt{}_rate{}_ep{}.pth'.format(
                               datetime.date.today(), args.img_size, args.num_classes, args.learning_rate,
                               args.batch_size,
                               args.dice_ce_rate,
                               args.epochs)))
                logging.info(f'Checkpoint {epoch} saved!')
                with open('./log.md', 'a') as f:
                    f.write(
                        f'best loss: {best_loss}, dice: {test_dice}, learning_rate:{args.learning_rate}, batchsize:{args.batch_size}, dice_ce_rate:{args.dice_ce_rate}\tepoch: {epoch}\n')
        # tf-logs
        writer.add_scalar('train/loss', np.array(train_loss), epoch)
        writer.add_scalar('train/dice', np.array(train_dice.cpu()), epoch)
        writer.add_scalar('test/loss', np.array(test_loss.cpu()), epoch)
        writer.add_scalar('test/dice', np.array(test_dice.cpu()), epoch)
        writer.add_scalar('OA', np.array(OA), epoch)
        writer.add_scalar('AA', np.array(AA), epoch)
        writer.add_scalar('Kappa', np.array(Kappa), epoch)
        writer.add_scalar('acc0_核仁', np.array(acc0), epoch)
        writer.add_scalar('acc1_核膜', np.array(acc1), epoch)
        writer.add_scalar('acc2_细胞质', np.array(acc2), epoch)
        writer.add_scalar('acc3_细胞边界', np.array(acc3), epoch)
        writer.add_scalar('acc4_背景', np.array(acc4), epoch)

        model.eval()

    result_dir = args.result_dir
    np.save(os.path.join(result_dir, 'train_losss.npy'), np.array(train_losss))
    np.save(os.path.join(result_dir, 'train_dices.npy'), np.array(train_dices))
    np.save(os.path.join(result_dir, 'test_losss.npy'), np.array(test_losss))
    np.save(os.path.join(result_dir, 'test_dices.npy'), np.array(test_dices))
    np.save(os.path.join(result_dir, 'OAs.npy'), np.array(OAs))
    np.save(os.path.join(result_dir, 'AAs.npy'), np.array(AAs))
    np.save(os.path.join(result_dir, 'Kappas.npy'), np.array(Kappas))
    np.save(os.path.join(result_dir, 'acc0s.npy'), np.array(acc0s))
    np.save(os.path.join(result_dir, 'acc1s.npy'), np.array(acc1s))
    np.save(os.path.join(result_dir, 'acc2s.npy'), np.array(acc2s))
    np.save(os.path.join(result_dir, 'acc3s.npy'), np.array(acc3s))
    np.save(os.path.join(result_dir, 'acc4s.npy'), np.array(acc4s))
    torch.cuda.empty_cache()
    writer.close()
    with open('./log.md', 'a') as f:
        f.write(f'-----------------------------------------------------------------------------------\n')
