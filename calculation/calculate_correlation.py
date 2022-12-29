from tqdm.auto import tqdm
import logging
import os
import argparse
from collections import OrderedDict
import json

from models import *
from pruner import get_input_channel_norm
from similarity import similarity


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate distace/similarity'
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
    parser.add_argument('--correlation',
                        default=['cosine_sim', 'Pearson_sim',
                                 'Euclide_dis', 'Manhattan_dis', 'SNR_dis'
                                 ],
                        type=str,
                        nargs='+',
                        help='correlation'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/correlation_full_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_correlation_vgg16.txt',
                        help='path to log file',
                        )

    return parser


def get_correlation_mat(weight, criterion):
    """Caculate correlation matrix between channels in inter-layer
    """
    num_filters = weight.shape[1]
    correlation_mat = [[0.]*num_filters for i in range(num_filters)]

    # compute the channel pairwise similarity based on criterion
    for i in tqdm(range(num_filters)):
        for j in range(num_filters):
            filter_i = weight.detach()[:, i]
            filter_j = weight.detach()[:, j]
            correlation_mat[i][j] = similarity(
                filter_i, filter_j, criterion).item()

    return correlation_mat


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

    # calculate correlation matrix
    logging.info(f"--------------compute correlation directly--------------")
    correlation_dict = {}
    for criterion in args.correlation:
        logging.info(f"----------------{criterion}----------------")
        all_convs_cor_mat = {}
        for i_conv in range(len(all_convs) - 1):
            logging.info(f"------------i_conv: {i_conv}------------")
            next_conv = all_convs[i_conv + 1]
            correl_matrix = get_correlation_mat(next_conv.weight, criterion)
            all_convs_cor_mat[i_conv] = correl_matrix
        correlation_dict[criterion] = all_convs_cor_mat
    dict["correlation"] = {'full': correlation_dict}

    with open(args.output, 'w') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)
