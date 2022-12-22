from tqdm.auto import tqdm
import logging
import os
import argparse
from collections import OrderedDict
import json

from models import *
from decompose import decomposition


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate SVD+'
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
    parser.add_argument('--decomposer',
                        default=['hosvd'],
                        type=str,
                        nargs='+',
                        help='tensor decomposer'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/decomposition.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_decomposition.txt',
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

    # calculate decomposition
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
                u = decomposition(filter, decomposer)
                all_filters_u_dict[i_filter] = [i.tolist() for i in u]
            all_convs_dict[i_conv] = all_filters_u_dict
        dict[decomposer] = all_convs_dict

    with open(args.output, 'w') as file:
        json.dump(dict, file, indent=4)
