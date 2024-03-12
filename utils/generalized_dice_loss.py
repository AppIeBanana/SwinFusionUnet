import torch

def dice_coeff(input, target, epsilon=1e-6):
    # 计算单个样本的Dice系数
    inter = torch.sum(input * target, dim=(2, 3))
    sets_sum = torch.sum(input, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
    dice = (2 * inter + epsilon) / (sets_sum + epsilon)
    return dice.mean(dim=1)

def generalized_dice_loss(pred, target, weights=None, epsilon=1e-6):
    """
    计算通用化Dice损失

    Args:
        pred (Tensor): 预测结果张量，形状为 (B, C, H, W)
        target (Tensor): 真实标签张量，形状为 (B, C, H, W)
        weights (Tensor): 类别权重张量，形状为 (C,)，默认为均等权重
        epsilon (float): 避免分母为零的小常数

    Returns:
        Tensor: 通用化Dice损失值
    """
    assert pred.size() == target.size()
    print(pred.size(), target.size())
    if weights is None:
        weights = torch.ones(pred.size(1), device=pred.device)
    print(weights.size())
    dice = dice_coeff(pred, target, epsilon)

    numerator = 2 * torch.sum(weights * dice)
    denominator = torch.sum(weights * (torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target, dim=(0, 2, 3))))

    loss = 1 - (numerator + epsilon) / (denominator + epsilon)

    return loss