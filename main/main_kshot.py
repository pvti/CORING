
import os
import os.path as osp
import numpy as np
import time
import datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56, resnet_110
from models.cifar10.googlenet import googlenet, Inception
from models.cifar10.densenet import densenet_40

from data import cifar10
import utils.common as utils
import wandb

from rank import get_rank

parser = argparse.ArgumentParser("Cifar-10 iterativly pruning and finetuning")

parser.add_argument(
    '--data_dir',
    type=str,
    default='data',
    help='path to dataset')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    help='architecture')
parser.add_argument(
    '--job_dir',
    type=str,
    default='result',
    help='path for saving trained models')
parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')
parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='num of training epochs')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='init learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='50,100',
    type=str,
    help='learning rate')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-3,
    help='weight decay')
parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='checkpoint/cifar/cifar10/resnet_56.pt',
    help='pretrain model path')
parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9',
    help='compress rate of each conv')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--use_onecycle',
    dest='use_onecycle',
    action='store_true')
parser.add_argument(
    '--random_rank',
    dest='random_rank',
    action='store_true')
parser.add_argument(
    '--div_factor',
    type=float,
    default=10,
    help='div factor of OneCycle Learning rate Schedule (default: 10)')
parser.add_argument(
    '--final_div_factor',
    type=float,
    default=100,
    help='final div factor of OneCycle Learning rate Schedule (default: 100)')
parser.add_argument(
    '--pct_start',
    type=float,
    default=0.1,
    help='pct_start of OneCycle Learning rate Schedule (default: 0.1)')
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
parser.add_argument(
    '--shot',
    type=int,
    default=10,
    help='num of shot')
parser.add_argument(
    '--calib',
    type=int,
    default=100,
    help='num of calibrating epochs')

parser.set_defaults(use_onecycle=False)
parser.set_defaults(random_rank=False)

args = parser.parse_args()

# init wandb
name = args.criterion + '_' + args.compress_rate
if args.use_onecycle:
    if args.random_rank:
        name += '_random'
    else:
        name += '_onecycle'
else:
    if args.random_rank:
        name += '_random_small'
    else:
        name += '_standard'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = 10
print_freq = (256*50)//args.batch_size

args.job_dir = osp.join(args.job_dir, args.arch,
                        args.strategy, args.criterion,
                        str(args.shot), args.compress_rate)
if not osp.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(osp.join(args.job_dir, 'prune_finetune'+now+'.log'))

wandb.init(
    name=name,
    project='KShot' + '_' +
    args.job_dir.replace(args.criterion, '').replace('/', '_').replace(args.compress_rate, ''),
    config=vars(args)
)

# use for loading pretrain model
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''

# load training data
train_loader, val_loader = cifar10.load_data(args)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()


def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer

    cnt = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt += 1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info(f'computing rank for cov_id = {cov_id} ')
                rank = get_rank(oriweight, args.criterion, args.strategy)
                if args.random_rank:
                    rank = np.random.random_sample(rank.shape)
                select_index = np.argsort(
                    rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info(f'computing rank for cov_id = {cov_id} ')
                    rank = get_rank(oriweight, args.criterion, args.strategy)
                    if args.random_rank:
                        rank = np.random.random_sample(rank.shape)
                    # preserved filter id
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_google_model(model, oristate_dict):
    state_dict = model.state_dict()

    filters = [
        [64, 128, 32, 32],
        [128, 192, 96, 64],
        [192, 208, 48, 64],
        [160, 224, 64, 64],
        [128, 256, 64, 64],
        [112, 288, 64, 64],
        [256, 320, 128, 128],
        [256, 320, 128, 128],
        [384, 384, 128, 128]
    ]

    # last_select_index = []
    all_honey_conv_name = []
    all_honey_bn_name = []
    cur_last_select_index = []

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

            last_select_index = cur_last_select_index[:]
            cur_last_select_index = []

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

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[name_base+conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = list(range(0, orifilter_num))

                for i in range(state_dict[name_base+conv_name].size(0)):
                    for index_j, j in enumerate(select_index):
                        state_dict[name_base+conv_name][i][index_j] = \
                            oristate_dict[conv_name][i][j]

                if branch_name == '_n1x1':
                    tmp_select_index = list(
                        range(state_dict[name_base+conv_name].size(0)))
                    cur_last_select_index += tmp_select_index
                if branch_name == '_pool_planes':
                    tmp_select_index = list(
                        range(state_dict[name_base+conv_name].size(0)))
                    tmp_select_index = [x+filters[cov_id-2][0]+filters[cov_id-2]
                                        [1]+filters[cov_id-2][2] for x in tmp_select_index]
                    cur_last_select_index += tmp_select_index

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
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[name_base+conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info(
                        f'computing rank for name = {name} weight_index = {weight_index} conv_name = {conv_name}')
                    rank = get_rank(oriweight, args.criterion, args.strategy)
                    if args.random_rank:
                        rank = np.random.random_sample(rank.shape)
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                else:
                    select_index = list(range(0, orifilter_num))

                for index_i, i in enumerate(select_index):
                    state_dict[name_base+conv_name][index_i] = \
                        oristate_dict[conv_name][i]

                if branch_name == '_n3x3':
                    tmp_select_index = [x+filters[cov_id-2][0]
                                        for x in select_index]
                    cur_last_select_index += tmp_select_index
                if branch_name == '_n5x5':
                    last_select_index = select_index

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

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[name_base+conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = range(0, orifilter_num)

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                select_index_1 = copy.deepcopy(select_index)

                if orifilter_num != currentfilter_num:
                    logger.info(
                        f'computing rank for name = {name} weight_index = {weight_index} conv_name = {conv_name}')
                    rank = get_rank(oriweight, args.criterion, args.strategy)
                    if args.random_rank:
                        rank = np.random.random_sample(rank.shape)
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                else:
                    select_index = list(range(0, orifilter_num))

                if branch_name == '_n5x5':
                    tmp_select_index = [x+filters[cov_id-2][0] +
                                        filters[cov_id-2][1] for x in select_index]
                    cur_last_select_index += tmp_select_index

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                        state_dict[name_base+conv_name][index_i][index_j] = \
                            oristate_dict[conv_name][i][j]

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
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[name_base+conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info(
                        f'computing rank for name = {name} weight_index = {weight_index} conv_name = {conv_name}')
                    rank = get_rank(oriweight, args.criterion, args.strategy)
                    if args.random_rank:
                        rank = np.random.random_sample(rank.shape)
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    cur_last_select_index = select_index[:]

                    for index_i, i in enumerate(select_index):
                        state_dict[name_base+conv_name][index_i] = \
                            oristate_dict[conv_name][i]  # '''

    for name, module in model.named_modules():  # Reassign non sketch weights to the new network
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            if name not in all_honey_conv_name:
                state_dict[name_base+name +
                           '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base+name +
                           '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[name_base+name +
                           '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base+name +
                           '.bias'] = oristate_dict[name + '.bias']
                state_dict[name_base+name +
                           '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name_base+name +
                           '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_densenet_model(model, oristate_dict):

    state_dict = model.state_dict()
    last_select_index = []  # Conv index selected in the previous layer

    cnt = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt += 1
            cov_id = cnt
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                logger.info(f'computing rank for cov_id = {cov_id} ')
                rank = get_rank(oriweight, args.criterion, args.strategy)
                if args.random_rank:
                    rank = np.random.random_sample(rank.shape)
                # preserved filter id
                select_index = list(np.argsort(
                    rank)[orifilter_num-currentfilter_num:])
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            if j < len(oristate_dict[name + '.weight'][i]):
                                state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        if j < len(oristate_dict[name + '.weight'][i]):
                            state_dict[name_base+name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                select_index = list(range(0, orifilter_num))

            else:
                select_index = list(range(0, orifilter_num))
                state_dict[name_base+name + '.weight'] = oriweight

            if cov_id == 1 or cov_id == 14 or cov_id == 27:
                last_select_index = select_index
            else:
                tmp_select_index = [x+cov_id*12 -
                                    (cov_id-1)//13*12 for x in select_index]
                last_select_index += tmp_select_index

    model.load_state_dict(state_dict)


def get_cpr():
    if args.compress_rate:
        import re
        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

    cpr = [[]] * args.shot
    for shot in range(args.shot):
        cpr[shot] = [r / (args.shot - shot) for r in compress_rate]

    return cpr


def prune(input_model, compress_rate):
    # load target model architecture
    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    prune_model = eval(args.arch)(compress_rate=compress_rate).cuda()
    logger.info(prune_model)

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        prune_model = nn.DataParallel(prune_model, device_ids=device_id).cuda()

    inpstate_dict = input_model.state_dict()
    logger.info('Pruning model...')
    if args.arch == 'googlenet':
        load_google_model(prune_model, inpstate_dict)
    elif args.arch == 'vgg_16_bn':
        load_vgg_model(prune_model, inpstate_dict)
    elif args.arch == 'resnet_56':
        load_resnet_model(prune_model, inpstate_dict, 56)
    elif args.arch == 'resnet_110':
        load_resnet_model(prune_model, inpstate_dict, 110)
    elif args.arch == 'densenet_40':
        load_densenet_model(prune_model, inpstate_dict)
    else:
        raise ValueError("Not implemented arch")

    logger.info('Calibrating pruned model...')
    # fix this, args.lr_decay_step
    local_optimizer = torch.optim.SGD(prune_model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    local_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        local_optimizer, milestones=lr_decay_step, gamma=0.1)
    epoch = 0
    while epoch < math.ceil(args.calib/args.shot):
        train_obj, train_top1_acc,  train_top5_acc = train(
            epoch,  train_loader, prune_model, criterion, local_optimizer, local_scheduler)
        epoch += 1

    return prune_model


def main():

    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    # load baseline model
    logger.info('Loading baseline model')
    origin_model = eval(args.arch)(compress_rate=[0.] * 100).cuda()
    ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')

    if args.arch == 'densenet_40' or args.arch == 'resnet_110':
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt['state_dict'])

    cpr = get_cpr()
    # process k-shot pruning
    logger.info(f'Process {args.shot}-shot pruning:')
    model = copy.deepcopy(origin_model)
    for shot in range(args.shot):
        logger.info(f'Shot {shot}:')
        model = prune(model, cpr[shot])

    # evaluate model
    logger.info(f'Evaluating pruned model:')
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    _, pruned_acc, _ = validate(val_loader, model, criterion)
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))
    logger.info('Pruned accuracy: %.2f' % (pruned_acc))
    wandb.log({'params': params, 'flops': flops, 'pruned_acc': pruned_acc})

    # fine-tuning
    logger.info('Finetuning model:')
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_decay_step, gamma=0.1)
    if args.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, div_factor=args.div_factor, epochs=args.epochs, steps_per_epoch=len(
            train_loader), pct_start=args.pct_start, final_div_factor=args.final_div_factor)

    start_epoch = 0
    best_top1_acc = 0

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        if not args.use_onecycle:
            scheduler.step()
        else:
            for _ in range(len(train_loader)):
                scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(
            epoch,  train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(
            val_loader, model, criterion)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.job_dir)

        cur_lr = optimizer.param_groups[0]["lr"]
        wandb.log({'epoch': epoch, 'best_acc': max(valid_top1_acc, best_top1_acc), 'top1': valid_top1_acc,
                   'top5': valid_top5_acc, 'lr': cur_lr})

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))

    wandb.save(osp.join(args.job_dir, '*'))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.use_onecycle:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # cur lr
        cur_lr = next(iter(optimizer.param_groups))['lr']

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '
                'Lr {cur_lr:.4f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5, cur_lr=cur_lr))

    if not args.use_onecycle:
        scheduler.step()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
