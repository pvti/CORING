from tqdm.auto import tqdm
import logging
import os
import argparse
from collections import OrderedDict
import json
import numpy as np
import torch

from models import *


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate norm'
                                     )
    parser.add_argument('--arch',
                        default='VGG16',
                        help='network architecture',
                        )
    parser.add_argument('--checkpoint',
                        default='checkpoint/baseline/vgg/vgg16.pth',
                        metavar='FILE',
                        help='path to load checkpoint',
                        )
    parser.add_argument('--norm',
                        default=['L0', 'L1', 'L2', 'inf'
                                 ],
                        type=str,
                        nargs='+',
                        help='Lp norm'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/norm_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_norm_vgg16.txt',
                        help='path to log file',
                        )

    return parser


def get_norm_arr(weight, norm):
    """Caculate norm array of all filters in inter-layer
    """
    in_channels = weight.shape[1]
    norm_arr = []

    for i_c in range(in_channels):
        channel_weight = weight.detach()[:, i_c]
        value = 0
        if (norm == 'L0'):
            value = torch.linalg.vector_norm(
                torch.flatten(channel_weight), 0)

        elif (norm == 'L1'):
            value = torch.linalg.vector_norm(
                torch.flatten(channel_weight), 1)

        elif (norm == 'L2'):
            value = torch.linalg.vector_norm(channel_weight)

        elif (norm == 'inf'):
            value = torch.linalg.vector_norm(
                torch.flatten(channel_weight), float('inf'))

        norm_arr.append(value.item())

    return norm_arr


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log_file),
                                  logging.StreamHandler()
                                  ]
                        )
    logging.info("Arguments: " + str(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = None
    if 'ResNet' in args.arch:
        net = getResNet(args.arch)
    elif 'VGG' in args.arch:
        net = VGG(args.arch)
    net = net.to(device)

    assert os.path.isfile(args.checkpoint), "Checkpoint not found!"
    logging.info(f"=> loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint)
    # load net without DataParallel
    net_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        net_state_dict[name] = v
    net.load_state_dict(net_state_dict)
    all_convs = [m for m in net.backbone if isinstance(m, nn.Conv2d)]

    dict = {
        "arch": args.arch,
        "checkpoint": args.checkpoint
    }

    # calculate norm array and rank
    logging.info(f"--------------compute norm array--------------")
    norm_dict = {}
    sort_idx_dict = {}
    for norm in args.norm:
        logging.info(f"----------------{norm}----------------")
        all_convs_norm = {}
        all_convs_sort_idx = {}
        for i_conv in tqdm(range(len(all_convs) - 1)):
            next_conv = all_convs[i_conv + 1]
            norm_arr = get_norm_arr(next_conv.weight, norm)
            all_convs_norm[i_conv] = norm_arr
            arr = np.array(norm_arr)
            sort_idx = arr.argsort(kind='stable')[::-1][:len(arr)]
            all_convs_sort_idx[i_conv] = sort_idx.tolist()
        norm_dict[norm] = all_convs_norm
        sort_idx_dict[norm] = all_convs_sort_idx
    dict["norm"] = norm_dict
    dict["sort_idx"] = sort_idx_dict

    with open(args.output, 'w+') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)
