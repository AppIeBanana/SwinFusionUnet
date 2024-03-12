import argparse
import datetime
import os
import random
import numpy as np
import torch
import itertools
from pathlib import Path
from torch import nn
from train import train_fusion
from train_single import train_single
from models.SwinFUnet import SwinTransformerFusion
from models.SwinTransformer import SwinTransformerSys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--in_chans_x', type=int, default=8)
parser.add_argument('--in_chans_y', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--amp', type=bool, default=True, help='if speed calculate')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='if save checkpoints')
parser.add_argument('--checkpoint_dir', type=str, default=Path('checkpoints'))
parser.add_argument('--log_dir_root', type=str, default=Path('tf-logs'))
parser.add_argument('--result_dir_root', type=str, default=Path('results'))
parser.add_argument('--seed', type=int, default=3401, help='random seed')


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # CPU
    torch.cuda.manual_seed_all(args.seed)  # GPU
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    # model = SwinTransformerFusion(img_size=args.img_size, patch_size=4, in_chans_x=args.in_chans_x,
    #                               in_chans_y=args.in_chans_y,
    #                               num_classes=args.num_classes,
    #                               embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
    #                               num_heads=[3, 6, 12, 24],
    #                               window_size=args.img_size // 32, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                               norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #                               use_checkpoint=False, final_upsample='expand_first')

    model = SwinTransformerSys(img_size=args.img_size,
                               patch_size=4,
                               in_chans=args.in_chans_x,
                               num_classes=args.num_classes,
                               embed_dim=96,
                               depths=[2, 2, 2, 2],
                               depths_decoder=[1, 2, 2, 2],
                               num_heads=[3, 6, 12, 24],
                               window_size=args.img_size // 32,
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

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    learning_rates = [1e-3, 1e-4, 2e-4]
    batch_sizes = [8, 16]
    epochs = [50, 100]
    dice_ce_rate = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    param_combinations = itertools.product(learning_rates, batch_sizes, epochs, dice_ce_rate)
    for params in param_combinations:
        learning_rate, batch_size, epochs, dice_ce_rate = params

        log_dir_new = '{}_bt{}_lr{}_ep{}_dr{}_img{}'.format(datetime.date.today(), batch_size, learning_rate, epochs,
                                                            dice_ce_rate, args.img_size)
        log_dir = os.path.join(args.log_dir_root, log_dir_new)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        result_dir_new = '{}_bt{}_lr{}_ep{}_dr{}_img{}'.format(datetime.date.today(), batch_size, learning_rate, epochs,
                                                               dice_ce_rate, args.img_size)
        result_dir = os.path.join(args.result_dir_root, result_dir_new)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        train_single(args, model, device,log_dir,result_dir,learning_rate,batch_size,epochs, dice_ce_rate)
        # train_fusion(args, model, device,log_dir,result_dir,learning_rate,batch_size,epochs, dice_ce_rate)
