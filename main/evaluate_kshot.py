
import os
import os.path as osp
import numpy as np
import time
import datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
import wandb

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from models.imagenet.resnet import resnet_50
from models.imagenet.mobilenetv1 import mobilenet_v1
from models.imagenet.mobilenetv2 import mobilenet_v2

from data import imagenet
import utils.common as utils

from rank import get_rank

parser = argparse.ArgumentParser("ImageNet training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='data/imagenet',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
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
    default=200,
    help='num of training epochs')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='init learning rate')

parser.add_argument(
    '--lr_type',
    default='cos',
    type=str,
    help='learning rate decay schedule')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.99,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='weight decay')

parser.add_argument(
    '--label_smooth',
    type=float,
    default=0.1,
    help='label smoothing')

parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')

parser.add_argument(
    '--use_pretrain',
    action='store_true',
    help='whether use pretrain model')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='pretrain model path')

parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='rank',
    help='rank conv file folder')

parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')

parser.add_argument(
    '--test_only',
    action='store_true',
    help='whether it is test mode')

parser.add_argument(
    '--test_model_dir',
    type=str,
    default='',
    help='test model path')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--criterion',
    default='VBD_dis',
    type=str,
    help='criterion'
)

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
    default=10,
    help='num of calibrating epochs')

args = parser.parse_args()
if args.shot == 1:
    args.calib = 0

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CLASSES = 1000
print_freq = 128000//args.batch_size

args.job_dir = osp.join(args.job_dir, args.arch,
                        args.strategy, args.criterion,
                        str(args.shot), args.compress_rate)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

# save old training file
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
cp_file_dir = os.path.join(args.job_dir, 'cp_file/' + now)
if os.path.exists(args.job_dir+'/model_best.pth.tar'):
    if not os.path.isdir(cp_file_dir):
        os.makedirs(cp_file_dir)
    shutil.copy(args.job_dir+'/config.txt', cp_file_dir)
    shutil.copy(args.job_dir+'/logger.log', cp_file_dir)
    shutil.copy(args.job_dir+'/model_best.pth.tar', cp_file_dir)
    shutil.copy(args.job_dir + '/checkpoint.pth.tar', cp_file_dir)

# whether to override the logger file
if not args.resume:
    if os.path.exists(args.job_dir+'/logger.log'):
        os.remove(args.job_dir+'/logger.log')

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(
    args.job_dir, 'prune_finetune'+now+'.log'))

# init wandb
name = args.criterion + '_' + args.compress_rate
wandb.init(
    name=name,
    project='CriteriaComparison' + '_' +
    args.job_dir.replace(args.criterion, '').replace(
        '/', '_').replace(args.compress_rate, ''),
    config=vars(args)
)

# use for loading pretrain model
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = utils.CrossEntropyLabelSmooth(
    CLASSES, args.label_smooth)
criterion_smooth = criterion_smooth.cuda()

# load training data
print('==> Preparing data..')
data_tmp = imagenet.Data(args)
train_loader = data_tmp.train_loader
val_loader = data_tmp.test_loader

def load_resnet_model(model, oristate_dict):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet_50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    state_dict = model.state_dict()

    current_cfg = cfg[args.arch]
    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name = ['.weight', '.bias', '.running_mean',
                    '.running_var']  # ,'.num_batches_tracked']
    cnt = 1

    conv_weight_name = 'conv1.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[name_base+conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        logger.info(f'computing rank for cnt = {cnt}')
        rank = get_rank(oriweight, criterion=args.criterion, strategy=args.strategy)
        select_index = np.argsort(
            rank)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()

        for index_i, i in enumerate(select_index):
            state_dict[name_base+conv_weight_name][index_i] = \
                oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict[name_base + 'bn1' + bn_part][index_i] = \
                    oristate_dict['bn1' + bn_part][i]

        last_select_index = select_index
    else:
        state_dict[name_base + conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict[name_base + 'bn1' +
                       bn_part] = oristate_dict['bn1'+bn_part]

    state_dict[name_base + 'bn1' +
               '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

    cnt += 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            if args.arch == 'resnet_18' or args.arch == 'resnet_34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            if k == 0:
                iter += 1
            for l in range(iter):
                record_last = True
                if k == 0 and l == 2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    record_last = False
                elif k == 0 and l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info(f'computing rank for cnt = {cnt}')
                    rank = get_rank(oriweight, criterion=args.criterion, strategy=args.strategy)
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    if record_last:
                        last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]

                    if record_last:
                        last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if record_last:
                        last_select_index = None

                state_dict[name_base + bn_name +
                           '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
                cnt += 1

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_mobilenetv2_model(model, oristate_dict):

    state_dict = model.state_dict()

    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name = ['.weight', '.bias', '.running_mean', '.running_var']

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
                bn_name = block_name + str(l+1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info(f'computing rank for conv_cnt = {conv_cnt}')
                    rank = get_rank(oriweight, criterion=args.criterion, strategy=args.strategy)
                    select_index = np.argsort(
                        rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if (l == 6 or (l == 0 and layer_cnt != 1) or (l == 3 and layer_cnt == 1)) and last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]
                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    last_select_index = select_index

                elif (l == 6 or (l == 0 and layer_cnt != 1) or (l == 3 and layer_cnt == 1)) and last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    last_select_index = None

                state_dict[name_base + bn_name +
                           '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']

            layer_cnt += 1

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            bn_name = list(name[:])
            bn_name[-1] = str(int(name[-1])+1)
            bn_name = ''.join(bn_name)
            if conv_name not in all_honey_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]
                for bn_part in bn_part_name:
                    state_dict[name_base + bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]
                state_dict[name_base + bn_name +
                           '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_mobilenetv1_model(model, oristate_dict):

    state_dict = model.state_dict()

    last_select_index = None
    all_honey_conv_weight = []

    bn_part_name = ['.weight', '.bias', '.running_mean', '.running_var']
    prefix = args.rank_conv_prefix+'/rank_conv'
    subfix = ".npy"

    conv_cnt = 1
    for layer_cnt in range(13):
        conv_id = [0, 3]
        block_name = 'features.'+str(layer_cnt)+'.'
        for l in conv_id:
            conv_cnt += 1
            conv_name = block_name+str(l)
            bn_name = block_name + str(l + 1)

            conv_weight_name = conv_name + '.weight'
            all_honey_conv_weight.append(conv_weight_name)
            oriweight = oristate_dict[conv_weight_name]
            curweight = state_dict[name_base+conv_weight_name]
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                logger.info('loading rank from: ' +
                            prefix + str(conv_cnt) + subfix)
                rank = np.load(prefix + str(conv_cnt) + subfix)
                select_index = np.argsort(
                    rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                select_index.sort()

                if l == 3 and last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][i][j]
                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name + bn_part][index_i] = \
                                oristate_dict[bn_name + bn_part][i]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name_base+conv_weight_name][index_i] = \
                            oristate_dict[conv_weight_name][i]
                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name + bn_part][index_i] = \
                                oristate_dict[bn_name + bn_part][i]

                last_select_index = select_index

            elif l == 3 and last_select_index is not None:
                for index_i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+conv_weight_name][index_i][index_j] = \
                            oristate_dict[conv_weight_name][index_i][j]
                for bn_part in bn_part_name:
                    state_dict[name_base + bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]
                last_select_index = None

            else:
                state_dict[name_base+conv_weight_name] = oriweight
                for bn_part in bn_part_name:
                    state_dict[name_base + bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]
                last_select_index = None

            state_dict[name_base + bn_name +
                       '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            bn_name = list(name[:])
            bn_name[-1] = str(int(name[-1]) + 1)
            bn_name = ''.join(bn_name)
            if conv_name not in all_honey_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]
                for bn_part in bn_part_name:
                    state_dict[name_base + bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, epoch, step, len_iter):

    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.1 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.learning_rate * \
            (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    if step == 0:
        logger.info('learning_rate: ' + str(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def prune(origin_model, compress_rate):
    # load target model architecture
    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    model = eval(args.arch)(compress_rate=compress_rate).cuda()
    logger.info(model)

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    oristate_dict = origin_model.state_dict()
    logger.info('Pruning model...')
    if args.arch == 'resnet_50':
        load_resnet_model(model, oristate_dict)
    elif args.arch == 'mobilenet_v2':
        load_mobilenetv2_model(model, oristate_dict)
    elif args.arch == 'mobilenet_v1':
        load_mobilenetv1_model(model, oristate_dict)
    else:
        raise ValueError("Not implemented arch")

    logger.info('Calibrating pruned model...')
    # fix this, args.lr_decay_step
    local_optimizer = torch.optim.SGD(model.parameters(),
                                      lr=args.learning_rate,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    epoch = 0
    while epoch < math.ceil(args.calib/args.shot):
        train_obj, train_top1_acc,  train_top5_acc = train(
            epoch,  train_loader, model, criterion_smooth, local_optimizer)
        epoch += 1

    return model


def main():

    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    cpr = get_cpr()
    
    start_epoch = 0
    best_top1_acc = 0
    best_top5_acc = 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.job_dir, 'checkpoint.pth.tar')
    if args.resume:
        model = eval(args.arch)(compress_rate=cpr[args.shot]).cuda()
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch']+1
        best_top1_acc = checkpoint['best_top1_acc']
        if 'best_top5_acc' in checkpoint:
            best_top5_acc = checkpoint['best_top5_acc']

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']
        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict, strict=False)
        logger.info("loaded checkpoint {} epoch = {}".format(
            checkpoint_dir, checkpoint['epoch']))
    else:
        if args.use_pretrain:
            # load baseline model
            logger.info('Loading baseline model')
            origin_model = eval(args.arch)(compress_rate=[0.] * 100).cuda()
            ckpt = torch.load(args.pretrain_dir)
            origin_model.load_state_dict(ckpt['state_dict'], strict=False)

            # process k-shot pruning
            logger.info(f'Process {args.shot}-shot pruning:')
            model = origin_model
            for shot in range(args.shot):
                logger.info(f'Shot {shot}:')
                model = prune(model, cpr[shot])

    # evaluate model
    _, pruned_acc, _ = validate(val_loader, model, criterion)
    logger.info('Pruned accuracy: %.2f' % (pruned_acc))
    wandb.log({'pruned_acc': pruned_acc})

    # train the model
    optimizer = torch.optim.SGD(model.parameters(
    ), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(
            epoch,  train_loader, model, criterion_smooth, optimizer)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(
            val_loader, model, criterion)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.job_dir)

        cur_lr = optimizer.param_groups[0]["lr"]
        wandb.log({'epoch': epoch, 'best_acc': max(valid_top1_acc, best_top1_acc), 'top1': valid_top1_acc,
                   'top5': valid_top5_acc, 'lr': cur_lr})

        epoch += 1
        logger.info("=>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(
            best_top1_acc, best_top5_acc))

    wandb.save(osp.join(args.job_dir, '*'))

    training_time = (time.time() - start_t) / 36000
    logger.info('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()
    # scheduler.step()

    num_iter = len(train_loader)

    print_freq = num_iter // 10

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
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
