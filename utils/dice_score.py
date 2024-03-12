from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn

from .conf_matrix import *


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # Dice系数作为分割损失函数。
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:  # 如果仅有两个维度,且需要reduce_batch_first,则报错。否则计算每个batch样本分数后平均,或者直接压平计算
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    # 计算单个batch的交集和、并集和
    # input: C L
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # 如果输入与目标形状一致,直接用torch.dot计算交集（乘积和）
        sets_sum = torch.sum(input) + torch.sum(target)  # 求并集是输入与目标各自和的和(加法和)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        # input: B C L
        dice = 0
        for i in range(input.shape[0]):  # 计算每个batch的交集和并集，并求和
            dice += dice_coeff(input[i, ...], target[i, ...])  # 分子为输入与目标的交集体素点总和，分母为输入与目标的和加上一个很小的ε值,避免分母为0
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # 分别计算单个通道dice再取平均
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def single_dice_score(y_, y, num_classes):
    dice_score = 0
    if num_classes == 1:
        y_ = (F.sigmoid(y_) > 0.5).float()
        dice_score += dice_coeff(y_, y, reduce_batch_first=False)
    else:
        y_ = F.one_hot(y_.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
        y = F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()
        dice_score += multiclass_dice_coeff(y_[:, 1:, ...], y[:, 1:, ...], reduce_batch_first=False)
    return dice_score


def multi_dice_score(net, dataloader, device, dice_ce_rate, experiment=None, optimizer=None, batch_size=2,
                     num_classes=5):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score, loss, all_lab, all_pre = 0, 0, [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for (x, y, label) in tqdm(dataloader, desc='Validation round'):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            with torch.no_grad():
                out = net(x, y)
                loss += (1 - dice_ce_rate) * criterion(out, label) + dice_ce_rate * dice_loss(
                    F.softmax(out, dim=1).float(), F.one_hot(label, num_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True)
                # loss += 1 * criterion(out, label) + 5 * dice_loss(F.softmax(out, dim=1).float(),
                #                                              F.one_hot(label, num_classes).permute(0, 3, 1, 2).float(),
                #                                              multiclass=True)
                single_test_dice = single_dice_score(out, label, num_classes)
                dice_score += single_test_dice

                all_lab.append(label.cpu().numpy())
                out = nn.functional.softmax(out, 1).argmax(1)
                all_pre.append(out.cpu().numpy())
                # experiment.log({
                #     'learning rate': optimizer.param_groups[0]['lr'],
                #     'test Dice': single_test_dice,
                # })

    all_Y = np.concatenate(all_lab)
    # print("True lbl:", np.unique(all_Y))
    all_Y_ = np.concatenate(all_pre)
    # print("Pre lbl:", np.unique(all_Y_))
    every_class, confusion_mat = cal_Accuracy(all_Y, all_Y_, num_classes)
    acc = every_class[:]
    # acc = [0 for i in range(10)]
    net.train()
    if num_val_batches == 0:
        return dice_score
    if num_classes == 2:
        return dice_score / num_val_batches, loss / num_val_batches, acc[num_classes], acc[num_classes + 1], acc[
            num_classes + 2], \
            acc[0], acc[1]
    if num_classes == 7:
        return dice_score / num_val_batches, loss / num_val_batches, acc[num_classes], acc[num_classes + 1], acc[
            num_classes + 2], \
            acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]
    if num_classes == 5:
        return dice_score / num_val_batches, loss / num_val_batches, acc[num_classes], acc[num_classes + 1], acc[
            num_classes + 2], \
            acc[0], acc[1], acc[2], acc[3], acc[4]
    if num_classes == 4:
        return dice_score / num_val_batches, loss / num_val_batches, acc[num_classes], acc[num_classes + 1], acc[
            num_classes + 2], \
            acc[0], acc[1], acc[2], acc[3]
