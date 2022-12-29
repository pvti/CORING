from tqdm.auto import tqdm
import logging
import os
import argparse
from collections import OrderedDict
import json
import torch

from models import *
from .decompose import decompose


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate SVD+'
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
    parser.add_argument('--decomposer',
                        default=['svd', 'hosvd'],
                        type=str,
                        nargs='+',
                        help='tensor decomposer'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/decomposition_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_decomposition_vgg16.txt',
                        help='path to log file',
                        )

    return parser


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

    # calculate decomposition
    decomposition_dict = {}
    for decomposer in args.decomposer:
        logging.info(f"------------------{decomposer}------------------")
        all_convs_dict = {}
        for i_conv in range(len(all_convs) - 1):
            logging.info(f"------------------{i_conv}------------------")
            next_conv = all_convs[i_conv + 1]
            weight = next_conv.weight
            num_filters = weight.shape[1]
            all_filters_u_dict = {}
            for i_filter in tqdm(range(num_filters)):
                filter = weight.detach()[:, i_filter]
                u = decompose(filter, decomposer)
                all_filters_u_dict[i_filter] = [i.tolist() for i in u]
            all_convs_dict[i_conv] = all_filters_u_dict

        decomposition_dict[decomposer] = all_convs_dict

    dict['decomposition'] = decomposition_dict

    with open(args.output, 'w') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)
