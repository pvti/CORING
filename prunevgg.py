import argparse
import collections
import logging
import os
import json
import torch

from data import DataLoaderCIFAR10
from helpers import get_model_performance
from models import VGG
from pruner import apply_channel_sorting, channel_prune
from config.default import pruning_types, pruning_cfg
from train import evaluate


def get_parser():
    """Define the command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description='VGG16 Similarity based filter pruning')
    parser.add_argument('--checkpoint', default='checkpoint/baseline/vgg/vgg16.pth', metavar='FILE',
                        help='path to load checkpoint')
    parser.add_argument('--calculated_rank', default='calculation/baseline/vgg/rank_vgg16.json',
                        help='path to load pre-calculated rank file')
    parser.add_argument('--decomposer', default=['full', 'svd', 'hosvd'], type=str, nargs='+',
                        help='tensor decomposer')
    parser.add_argument('--criteria', default=['cosine_sim', 'Pearson_sim', 'Euclide_dis', 'Manhattan_dis', 'SNR_dis'],
                        type=str, nargs='+', help='criteria')
    parser.add_argument('--strategy', default=['sum', 'min_sum', 'min_min'], type=str, nargs='+',
                        help='filter ranking strategy')
    parser.add_argument('--type', default=['L1-A', 'L1-B', 'L1-C', 'Hrank-1', 'Hrank-2', 'Hrank-3', 'Acc', 'Pfm', 'Mdw'],
                        type=str, nargs='+', help='pruning type of the model')
    parser.add_argument('--output', default='checkpoint/pruned/vgg16/',
                        help='path to save checkpoint')
    parser.add_argument('--log_file', default='logs/log_prunevgg.txt',
                        help='path to log file')
    parser.add_argument('--device', default=0, type=int,
                        help='device to use for training')

    return parser


def main(args):
    """Main function for the script."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])
    logging.info("Arguments: " + str(args))

    # Load the CIFAR10 dataset and define the loss function
    cifar10_dataset = DataLoaderCIFAR10()
    dataloader = cifar10_dataset.dataloader
    criterion = torch.nn.CrossEntropyLoss()

    # Set the device to use for training
    device = 'cuda:{}'.format(
        args.device) if torch.cuda.is_available() else 'cpu'

    # Load the VGG16 model
    net = VGG('VGG16')
    net = net.to(device)

    # Load the state dictionary for the pre-trained model from the checkpoint file
    assert os.path.isfile(args.checkpoint), "Checkpoint not found!"
    logging.info(f"=> Loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint)
    # Load net without DataParallel
    net_state_dict = collections.OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        net_state_dict[name] = v
    net.load_state_dict(net_state_dict)
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    logging.info(f'Baseline model: best_acc (%) = {best_acc} size (Mb), latency (ms), macs (M), num_params (M) = '
                 f'{get_model_performance(net)}')

    # Load the pre-calculated sort idx
    logging.info(
        f"=> Loading pre-calculated rank file '{args.calculated_rank}'")
    with open(args.calculated_rank, 'r') as fp:
        rank = json.load(fp)

    # measure performance of all type
    logging.info(f"=> Measuring performance of all type...")
    for prune_type in args.type:
        logging.info(f"Pruning type: {prune_type}")
        ratio = pruning_types[prune_type]
        rand_prune = channel_prune(net, ratio)
        logging.info(
            f"size (Mb), latency (ms), macs (M), num_params (M) = {get_model_performance(rand_prune)}")

    # Loop over the decomposition methods, ranking criteria, and ranking strategies
    os.makedirs(args.output, exist_ok=True)
    logging.info(f"=> Looping over all combinations...")
    pruned_acc_strategy = {s: {} for s in args.strategy}
    for strategy in args.strategy:
        for decomposer in args.decomposer:
            pruned_acc_dict = {c: [] for c in args.criteria}
            for criteria in args.criteria:
                logging.info(
                    f"Decomposer: {decomposer}, Criteria: {criteria}, Strategy: {strategy}")
                idx_sort = rank[strategy][decomposer]['sort_idx'][criteria]
                net_sorted = apply_channel_sorting(net, idx_sort)

                # Loop over the different pruning types
                for prune_type in args.type:
                    logging.info(f"Pruning type: {prune_type}")
                    ratio = pruning_types[prune_type]
                    net_pruned = channel_prune(net_sorted, ratio)

                    # Evaluate the pruned model
                    acc = evaluate(net_pruned, dataloader['test'],
                                   criterion, device)
                    logging.info(f"Pruned model: acc (%) = {acc:.2f}")
                    pruned_acc_dict[criteria].append(acc)

                    # Save the pruned model
                    state = {
                        'cfg': pruning_cfg[prune_type],
                        'net': net_pruned.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                        'decomposer': decomposer,
                        'criteria': criteria,
                        'strategy': strategy,
                        'type': prune_type
                    }
                    path = os.path.join(
                        args.output, f"vgg16_{decomposer}_{criteria}_{strategy}_{prune_type}.pth")
                    logging.info(f"Saving pruned model to {path}")
                    torch.save(state, path)

            pruned_acc_strategy[strategy][decomposer] = pruned_acc_dict
    logging.info(pruned_acc_strategy)


if __name__ == "__main__":
    main(get_parser().parse_args())
