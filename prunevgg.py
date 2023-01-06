import logging
import os
import argparse
from collections import OrderedDict
import json
import torch
import torch.nn as nn

from data import DataLoaderCIFAR10
from helpers import get_model_performance
from models import VGG
from pruner import channel_prune, apply_channel_sorting
from train import evaluate
from miscellaneous.plot import plot_decomposer


def get_parser():
    parser = argparse.ArgumentParser(description='VGG16 Similarity based filter pruning'
                                     )
    parser.add_argument('--checkpoint',
                        default='checkpoint/baseline/vgg/vgg16.pth',
                        metavar='FILE',
                        help='path to load checkpoint'
                        )
    parser.add_argument('--calculated_rank',
                        default='calculation/baseline/vgg/rank_vgg16.json',
                        help='path to load pre-calculated rank file'
                        )                        
    parser.add_argument('--decomposer',
                        default=['full', 'svd', 'hosvd'],
                        type=str,
                        nargs='+',
                        help='tensor decomposer'
                        )                        
    parser.add_argument('--criteria',
                        default=['cosine_sim', 'Pearson_sim',
                                 'Euclide_dis', 'Manhattan_dis', 'SNR_dis'],
                        type=str,
                        nargs='+',
                        help='criteria'
                        )
    parser.add_argument('--strategy',
                        default=['sum', 'min_sum', 'min_min'],
                        type=str,
                        nargs='+',
                        help='filter ranking strategy'
                        )
    parser.add_argument('--type',
                        default=['L1-A', 'L1-B', 'L1-C',
                                 'Hrank-1', 'Hrank-2', 'Hrank-3',
                                 'Acc', 'Pfm', 'Mdw'],
                        type=str,
                        nargs='+',
                        help='pruning type of the model'
                        )
    parser.add_argument('--output',
                        default='checkpoint/pruned/vgg16/',
                        help='path to save checkpoint'
                        )
    parser.add_argument('--log_file',
                        default='logs/log_prunevgg.txt',
                        help='path to log file'
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

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0

    net = VGG('VGG16')
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
    best_acc = checkpoint['acc']
    logging.info(
        f'Baseline model: best_acc (%) = {best_acc} size (Mb), latency (ms), macs (M), num_params (M) = {get_model_performance(net)}')

    # load pre-calculated sort idx
    logging.info(f"=> loading '{args.calculated_rank}'")
    f = open(args.calculated_rank, 'r')
    data = json.loads(f.read())

    # @TODO move this to config
    # set-up pruning type
    r1 = 1 - 196./256
    r2 = 1 - 196./512
    type_list = {
        'L1-A': [0.5, 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5],
        'L1-B': [0.5, 0., 0., 0., r1, r1, r1, r2, r2, 0.5, 0.5, 0.5],
        'L1-C': [0.5, 0., 0.5, 0., 0.5, 0.5, 0.5, r2, r2, r2, r2, r2],
        'Hrank-1': [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.75, 0.75, 0.75, 0.75, 0.75],
        'Hrank-2': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.75, 0.75, 0.75, 0.75, 0.75],
        'Hrank-3': [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.78, 0.78, 0.78, 0.78, 0.78],
        'Acc': [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.78, 0.78, 0.78, 0.78, 0.78],
        'Pfm': [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.78, 0.78, 0.78, 0.78, 0.78],
        'Mdw': [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.78, 0.78, 0.78, 0.78, 0.78],
    }
    # measure performance of all type
    logging.info(f"=> measuring performance of all type...")
    for type in args.type:
        logging.info(f"---------------{type}---------------")
        ratio = type_list[type]
        rand_prune = channel_prune(net, ratio)
        logging.info(
            f"size (Mb), latency (ms), macs (M), num_params (M) = {get_model_performance(rand_prune)}")

    logging.info(f"=> pruning all combination...")
    pruned_acc_strategy = {s: {} for s in args.strategy}
    for strategy in args.strategy:
        logging.info(f"------------------{strategy}------------------")
        for decomposer in args.decomposer:
            logging.info(f"------------------{decomposer}------------------")
            pruned_acc_dict = {c: [] for c in args.criteria}
            for criteria in args.criteria:
                logging.info(f"---------------{criteria}---------------")
                sort_idx_dict = data[strategy][decomposer]['sort_idx'][criteria]
                sorted_net = apply_channel_sorting(net, sort_idx_dict)
                for type in args.type:
                    logging.info(f"---------------{type}---------------")
                    ratio = type_list[type]
                    pruned_net = channel_prune(sorted_net, ratio)
                    pruned_acc = evaluate(pruned_net,
                                          dataloader['test'],
                                          criterion,
                                          device,
                                          verbose=False
                                          )
                    logging.info(f"pruned_net_acc (%) = {pruned_acc}")
                    pruned_acc_dict[criteria].append(pruned_acc)
                    # save pruned net
                    state = {'net': pruned_net.state_dict(),
                             'acc': pruned_acc,
                             'epoch': checkpoint['epoch'],
                             }
                    path_save_net = os.path.join(args.output,
                                                 f"{criteria}_{type}.pth")
                    logging.info(f"=> saving pruned model to {path_save_net}")
                    torch.save(state, path_save_net)

            pruned_acc_strategy[strategy][decomposer] = pruned_acc_dict

    logging.info(pruned_acc_strategy)

    # Plot decomposers comparison
    # for strategy, strategy_dict in pruned_acc_strategy.items():
    #     path = os.path.join('figures/decomposer/pruned/', strategy + '.png')
    #     logging.info(f"=> Saving figure '{path}'")
    #     # @TODO change the plotting function because the ratio is now of defined type, not range
    #     plot_decomposer(strategy_dict, type_list, path)
