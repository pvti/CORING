
import os
import os.path as osp
import time
import datetime
import torch
import argparse
from collections import OrderedDict

import torch.nn as nn
import torch.utils
import torch.utils.data.distributed

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56
from models.cifar10.googlenet import googlenet
from models.cifar10.densenet import densenet_40
from models.cifar10.mobilenetv2 import mobilenet_v2
from models.imagenet.resnet import resnet_50

from data import cifar10, imagenet
import utils.common as utils

from thop import profile
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser("Profile pruned model")

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'imagenet'],
    help='Name of dataset. [cifar10, imagenet]')

parser.add_argument(
    '--data_dir',
    type=str,
    default='data/cifar/cifar10',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40',
             'googlenet', 'mobilenet_v2', 'resnet_50'),
    help='architecture')

parser.add_argument(
    '--job_dir',
    type=str,
    default='result',
    help='path for saving trained models')

parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')

parser.add_argument(
    '--model_path',
    type=str,
    default='',
    help='test model path')

parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='batch size')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()


args.job_dir = osp.join(args.job_dir, args.arch)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(osp.join(args.job_dir, 'test_'+now+'.log'))


def main():
    logger.info("args = %s", args)

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

    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    model = eval(args.arch)(compress_rate=compress_rate).cuda()
    logger.info(model)

    val_loader = None
    img_size = 32
    if args.dataset == 'cifar10':
        _, val_loader = cifar10.load_data(args)
    else:
        val_loader = imagenet.Data(args).test_loader
        img_size = 224

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if os.path.isfile(args.model_path):
        logger.info('loading checkpoint {}'.format(args.model_path))
        checkpoint = torch.load(args.model_path)
        if args.arch == 'resnet_50':
            state_dict = torch.load(args.model_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        logger.info('evaluating model...')
        _, top1, top5 = validate(val_loader, model, criterion)

    else:
        logger.info('please specify a checkpoint file')

    input_image = torch.randn(1, 3, img_size, img_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    flops_pt, params_pt = get_model_complexity_info(model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False)
    params = min(params, params_pt)
    flops = min(flops, flops_pt)
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))


def validate(val_loader, model, criterion):

    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():

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

            logger.info(
                'Batch({0}/{1}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
