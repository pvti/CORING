from tqdm.auto import tqdm
import logging
import os
import argparse
from collections import OrderedDict
import json

from models import *
from pruner import get_input_channel_similarity, get_input_channel_norm


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate distace/similarity'
                                     )
    parser.add_argument('--net',
                        default='VGG19',
                        help='network type',
                        )
    parser.add_argument('--checkpoint',
                        default='checkpoint/baseline/vgg/vgg19.pth',
                        metavar='FILE',
                        help='path to load checkpoint',
                        )
    parser.add_argument('--norm',
                        default=['L0_norm', 'L1_norm', 'L2_norm', 'inf_norm'
                                 ],
                        type=str,
                        nargs='+',
                        help='Lp norm'
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
                        default='calculation/baseline/vgg/criteria.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_criteria.txt',
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
    if (args.net == "ResNet18"):
        net = ResNet18()
    elif 'VGG' in args.net:
        net = VGG(args.net)
    else:
        net = VGG('VGG19')
    net = net.to(device)

    assert os.path.isfile(args.checkpoint), f"{args.checkpoint} not found!"
    logging.info(f"=> loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint)
    # load net without DataParallel
    net_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        net_state_dict[name] = v
    net.load_state_dict(net_state_dict)

    dict = {
        "net": args.net,
        "checkpoint": args.checkpoint
    }

    all_convs = [m for m in net.backbone if isinstance(m, nn.Conv2d)]

    # calculate norm array
    norm_dict = {}
    for norm in args.norm:
        logging.info(f"------{norm}------")
        all_convs_norm = {}
        for i_conv in tqdm(range(len(all_convs) - 1)):
            next_conv = all_convs[i_conv + 1]
            norm_arr = get_input_channel_norm(next_conv.weight, norm)
            all_convs_norm[i_conv] = norm_arr.tolist()
        norm_dict[norm] = all_convs_norm
    dict["norm"] = norm_dict

    # calculate correlation matrix
    correlation_dict = {}
    for criterion in args.correlation:
        logging.info(f"------{criterion}------")
        all_convs_cor_mat = {}
        for i_conv in tqdm(range(len(all_convs) - 1)):
            next_conv = all_convs[i_conv + 1]
            correl_matrix = get_input_channel_similarity(next_conv.weight,
                                                         criterion)
            all_convs_cor_mat[i_conv] = correl_matrix.tolist()
        correlation_dict[criterion] = all_convs_cor_mat
    dict["correlation"] = correlation_dict

    with open(args.output, 'w') as file:
        json.dump(dict, file, indent=4)
