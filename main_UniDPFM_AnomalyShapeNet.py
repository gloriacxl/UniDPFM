import os

import torch
import torch.nn as nn
import numpy as np
import random
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
from collections import OrderedDict
from utils.tokenizer import SimpleTokenizer
import open_clip
from linear_origin import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble_origin import encode_text_with_prompt_ensemble
import models.ULIP_models as models
from data.AnomalyShapeNet import Dataset3dad_ShapeNet_train_final_newaug
from utils import utils


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    features_list = args.features_list
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    
    root_dir = './Anomaly-ShapeNet/pcd/'
    trainable_layer = LinearLayer(384, 512,
                                  len(args.features_list), args.model).to(device)

    optimizer = torch.optim.Adam(list(trainable_layer.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    with torch.amp.autocast(device_type='cuda'), torch.no_grad():
        obj_list = ['ashtray','bag','bottle','bowl',
                   'bucket','cap','cup','eraser',
                   'headset','helmet','jar','microphone','shelf','tap','vase']
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
    for epoch in range(epochs):
        class_loss = []

        for obj_class in[
                   'ashtray0','bag0',
                   'bottle0','bottle1','bottle3','bowl0','bowl1','bowl2','bowl3','bowl4','bowl5','bucket0','bucket1','cap0','cap3','cap4','cap5',
                   'cup0','cup1','eraser0','headset0','headset1','helmet0','helmet1','helmet2','helmet3','jar0','microphone0','shelf0',
                   'tap0','tap1','vase0','vase1','vase2','vase3','vase4','vase5','vase7','vase8','vase9']:
            train_dataloader = DataLoader(Dataset3dad_ShapeNet_train_final_newaug(root_dir, obj_class, 1024, True),
                                    batch_size=4, shuffle=True, drop_last=False)
            loss_list = []
            idx = 0
            for pc, mask, label, sample_path in train_dataloader:
                idx += 1
                pc = pc.cuda().to(torch.float32) 
                
                with torch.amp.autocast(device_type='cuda'):
                    with torch.no_grad():
                        pc_features,patch_tokens,center_idx = utils.get_model(model).encode_pc(pc)
                        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                        patch_tokens = [patch_tokens[j] for j in features_list]
                        
                    patch_tokens = trainable_layer(patch_tokens,obj_class)
                    cls = obj_class[:-1]
                    anomaly_maps = []
                    for layer in range(len(patch_tokens)):
                        patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)

                        anomaly_map = (100.0 * patch_tokens[layer] @ text_prompts[cls])
                        B, L, C = anomaly_map.shape

                        anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, L)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        anomaly_maps.append(anomaly_map)

                mask = mask.squeeze()
                gt = mask.to(device)
                gt_center = gt.gather(1, center_idx)
                loss = 0
                for num in range(len(anomaly_maps)):
                    loss = loss + loss_dice(anomaly_maps[num][:, 1, :], gt_center)
                    loss = loss + loss_focal(anomaly_maps[num], gt_center)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            if (epoch + 1) % args.print_freq == 0:
                logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

            class_loss.append(np.mean(loss_list))
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(class_loss)))

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + obj_class + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    parser.add_argument("--dataset", type=str, default='AnomalyShapeNet', help="train dataset name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3,7,11], help="features used")
    # hyper-parameter
    parser.add_argument("--save_path", type=str, default='./ckpt', help='path to save results')
    parser.add_argument("--epoch", type=int, default=700, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=100, help="save frequency")
    

    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    # Training

    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_ckpt_addr', default='./pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt', help='the ckpt to test 3d zero shot')
    parser.add_argument('--gpu', default='2', type=int, help='GPU id to use.')
    args = parser.parse_args()

    setup_seed(0)
    train(args)

