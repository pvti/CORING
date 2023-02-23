import os
import os.path as osp
import numpy as np
import datetime
import argparse
import copy
from collections import OrderedDict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56, resnet_110
from models.cifar10.googlenet import googlenet, Inception
from models.cifar10.densenet import densenet_40
from models.imagenet.resnet import resnet_50
#from models.cifar10.mobilenetv2 import mobilenet_v2
from models.imagenet.mobilenetv2 import mobilenet_v2

import utils.common as utils

import sys
sys.path.append(os.path.dirname('..'))
from similarity import similarity
from ranking_strategy import get_saliency
from decompose import decompose


parser = argparse.ArgumentParser("Rank generation")
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    help='architecture')
parser.add_argument(
    '--job_dir',
    type=str,
    default='result',
    help='path for saving ranking log')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='checkpoint/cifar/cifar10/resnet_56.pt',
    help='pretrain model path')
parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='rank',
    help='rank conv file folder')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')
parser.add_argument(
    '--criterion',
    default='VBD_dis',
    type=str,
    help='criterion')
parser.add_argument(
    '--strategy',
    default='min_sum',
    type=str,
    help='strategy')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.job_dir = osp.join(args.job_dir, args.arch, args.strategy, args.criterion)
if not osp.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(osp.join(args.job_dir, 'gen_rank_'+now+'.log'))

# use for loading pretrain model
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''

# creat folder to save rank files
prefix_folder = osp.join(args.rank_conv_prefix, args.arch,
                         args.strategy, args.criterion)
if not osp.isdir(prefix_folder):
    os.makedirs(prefix_folder)
prefix = osp.join(prefix_folder, 'rank_conv')
subfix = ".npy"


def get_correlation_mat(all_filters_u_dict):
    """Caculate correlation matrix between channels in inter-layer
        based on precalculated left-singular vectors dict of hosvd
    """
    num_filters = len(all_filters_u_dict)
    correlation_mat = [[0.]*num_filters for i in range(num_filters)]

    # compute the channel pairwise similarity based on criterion VBD_dis
    for i in tqdm(range(num_filters)):
        for j in range(num_filters):
            ui = all_filters_u_dict[i]
            uj = all_filters_u_dict[j]
            sum = 0.
            for x in range(3):
                sum += similarity(torch.tensor(ui[x]),
                                  torch.tensor(uj[x]),
                                  criterion=args.criterion
                                  )
            correlation_mat[i][j] = (sum/3).item()

    return correlation_mat


def get_rank(weight):
    """Get rank based on HOSVD+VBD+min_sum"""
    num_filters = weight.size(0)
    all_filters_u_dict = {}
    for i_filter in tqdm(range(num_filters)):
        filter = weight.detach()[i_filter, :]
        u = decompose(filter, decomposer='hosvd')
        all_filters_u_dict[i_filter] = [i.tolist() for i in u]

    correl_matrix = get_correlation_mat(all_filters_u_dict)
    correl_matrix = np.array(correl_matrix)

    dis = 1 if 'dis' in args.criterion else -1
    saliency = get_saliency(correl_matrix, args.strategy, dis)

    return saliency


def save(rank, cov_id, branch_name=''):
    save_pth = prefix + str(cov_id) + branch_name + subfix
    np.save(save_pth, rank)
    logger.info('rank saved to: ' + save_pth)


def rank_vgg(model, state_dict):
    # iterate over all convolutional layers
    cov_id = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            cov_id += 1
            weight = state_dict[name + '.weight']
            logger.info(f'=> calculating rank of: {name}')
            rank = get_rank(weight)
            save(rank, cov_id)


def rank_resnet(state_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    current_cfg = cfg[layer]

    # iterate over all convolutional layers
    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        print(f"layer_name = {layer_name}, num = {num}")
        for k in range(num):
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                print(f"conv_name = {conv_name}")
                conv_weight_name = conv_name + '.weight'
                weight = state_dict[conv_weight_name]
                logger.info(f'=> calculating rank of: {conv_weight_name}')
                rank = get_rank(weight)
                save(rank, cov_id)


def rank_googlenet(model, oristate_dict):
    all_honey_conv_name = []
    all_honey_bn_name = []

    cnt = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, Inception):
            cnt += 1
            cov_id = cnt

            honey_filter_channel_index = [
                '.branch5x5.6',
            ]  # the index of sketch filter and channel weight
            honey_channel_index = [
                '.branch1x1.0',
                '.branch3x3.0',
                '.branch5x5.0',
                '.branch_pool.1'
            ]  # the index of sketch channel weight
            honey_filter_index = [
                '.branch3x3.3',
                '.branch5x5.3',
            ]  # the index of sketch filter weight
            honey_bn_index = [
                '.branch3x3.4',
                '.branch5x5.4',
                '.branch5x5.7',
            ]  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_channel_index:
                if '3x3' in weight_index:
                    branch_name = '_n3x3'
                elif '5x5' in weight_index:
                    branch_name = '_n5x5'
                elif '1x1' in weight_index:
                    branch_name = '_n1x1'
                elif 'pool' in weight_index:
                    branch_name = '_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

            for weight_index in honey_filter_index:
                if '3x3' in weight_index:
                    branch_name = '_n3x3'
                elif '5x5' in weight_index:
                    branch_name = '_n5x5'
                elif '1x1' in weight_index:
                    branch_name = '_n1x1'
                elif 'pool' in weight_index:
                    branch_name = '_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                weight = oristate_dict[conv_name]
                logger.info(f'=> calculating rank of: {cov_id} {branch_name}')
                rank = get_rank(weight)
                save(rank, cov_id, branch_name)

            for weight_index in honey_filter_channel_index:
                if '3x3' in weight_index:
                    branch_name = '_n3x3'
                elif '5x5' in weight_index:
                    branch_name = '_n5x5'
                elif '1x1' in weight_index:
                    branch_name = '_n1x1'
                elif 'pool' in weight_index:
                    branch_name = '_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                weight = oristate_dict[conv_name]
                logger.info(f'=> calculating rank of: {cov_id} {branch_name}')
                rank = get_rank(weight)
                save(rank, cov_id, branch_name)

        elif name == 'pre_layers':
            cnt += 1
            cov_id = cnt
            honey_filter_index = ['.0']  # the index of sketch filter weight
            honey_bn_index = ['.1']  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_index:
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                weight = oristate_dict[conv_name]
                logger.info(f'=> calculating rank of: {cov_id}')
                rank = get_rank(weight)
                save(rank, cov_id)


def rank_densenet(model, state_dict):
    # iterate over all convolutional layers
    cov_id = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            cov_id += 1
            weight = state_dict[name + '.weight']
            logger.info(f'=> calculating rank of: {name}')
            rank = get_rank(weight)
            save(rank, cov_id)


def rank_resnet50(state_dict):
    cfg = {'resnet_50': [3, 4, 6, 3], }
    current_cfg = cfg[args.arch]

    cnt = 1
    conv_weight_name = 'conv1.weight'
    weight = state_dict[conv_weight_name]
    rank = get_rank(weight)
    save(rank, cnt)

    cnt += 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            iter = 3
            if k == 0:
                iter += 1
            for l in range(iter):
                if k == 0 and l == 2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                elif k == 0 and l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                weight = state_dict[conv_weight_name]
                logger.info(f'=> calculating rank of: {conv_weight_name}')
                rank = get_rank(weight)
                save(rank, cnt)
                cnt += 1


def rank_mobilenetv2(state_dict):
    layer_cnt = 1
    conv_cnt = 1
    cfg = [1, 2, 3, 4, 3, 3, 1, 1]
    for layer, num in enumerate(cfg):
        if layer_cnt == 1:
            conv_id = [0, 3]
        elif layer_cnt == 18:
            conv_id = [0]
        else:
            conv_id = [0, 3, 6]

        for k in range(num):
            if layer_cnt == 18:
                block_name = 'features.' + str(layer_cnt) + '.'
            else:
                block_name = 'features.'+str(layer_cnt)+'.conv.'

            for l in conv_id:
                conv_cnt += 1
                conv_name = block_name + str(l)

                conv_weight_name = conv_name + '.weight'
                weight = state_dict[conv_weight_name]
                logger.info(f'=> calculating rank of: {conv_weight_name}')
                rank = get_rank(weight)
                save(rank, conv_cnt)

            layer_cnt += 1


def main():
    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    logger.info(f'Loading pretrain model: {args.pretrain_dir}')
    model = eval(args.arch)(compress_rate=[0.] * 100).cuda()
    if args.arch == 'resnet_50':
        ckpt = torch.load(args.pretrain_dir)
        model.load_state_dict(ckpt)
        state_dict = model.state_dict()
        rank_resnet50(state_dict)
    else:
        ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')
        if args.arch == 'densenet_40' or args.arch == 'resnet_110':

            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(ckpt['state_dict'], strict=False)

        state_dict = model.state_dict()

        if args.arch == 'googlenet':
            rank_googlenet(model, state_dict)
        elif args.arch == 'vgg_16_bn':
            rank_vgg(model, state_dict)
        elif args.arch == 'resnet_56':
            rank_resnet(state_dict, 56)
        elif args.arch == 'resnet_110':
            rank_resnet(state_dict, 110)
        elif args.arch == 'densenet_40':
            rank_densenet(model, state_dict)
        elif args.arch == 'mobilenet_v2':
            rank_mobilenetv2(state_dict)
        else:
            raise ValueError("Not implemented arch")


if __name__ == '__main__':
    main()
