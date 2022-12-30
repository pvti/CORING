from tqdm.auto import tqdm
import os
import argparse
import logging
import matplotlib
from matplotlib import pyplot as plt
import torch
from collections import OrderedDict, defaultdict
import numpy as np
import math

from data import *
from models import *
from train import evaluate
from miscellaneous.prune_weight import fine_grained_prune

matplotlib.use('Agg')


def get_parser():
    parser = argparse.ArgumentParser(description='Scan model to view distribution'
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
    parser.add_argument('--output',
                        default='figures/baseline/vgg/',
                        help='path to folder to save figures',
                        )
    parser.add_argument('--log',
                        default='logs/log_scan.txt',
                        help='path to log file',
                        )

    return parser


def plot_weight_distribution(model, output, bins=256, count_nonzero_only=False):
    """Plot weight distribution
    """
    plot_size = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            plot_size += 1

    fig, axes = plt.subplots(round(plot_size/3), 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:  # bias layer has dim=1, conv has dim > 1
            logging.info(name)
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color='blue', alpha=0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    logging.info(f"=> Saving figure '{output}'")
    plt.savefig(output)


def plot_num_parameters_distribution(model, output):
    """Plot num parameters distribution
    """
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            logging.info(name)
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    logging.info(f"=> Saving figure '{output}'")
    plt.savefig(output)


def plot_num_filters_distribution(model, output):
    """Plot num filter distribution
    """
    num_filters = dict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            logging.info(name)
            num_filters[name] = layer.out_channels
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_filters.keys()), list(num_filters.values()))
    plt.title('#Filter Distribution')
    plt.ylabel('Number of Filter')
    plt.xticks(rotation=60)
    plt.tight_layout()
    logging.info(f"=> Saving figure '{output}'")
    plt.savefig(output)


@torch.no_grad()
def weight_sensitivity_scan(model,
                            dataloader,
                            scan_step=0.1,
                            scan_start=0.4,
                            scan_end=1.0,
                            verbose=True
                            ):
    """Scan weight sensitivity at each pruning step
    """
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param)
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, criterion, device)
            if verbose:
                print(
                    f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(
                f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
    return sparsities, accuracies


def plot_weight_sensitivity_scan(model, output):
    sparsities, accuracies = weight_sensitivity_scan(model,
                                                     dataloader['test'],
                                                     scan_step=0.1,
                                                     scan_start=0.4,
                                                     scan_end=1.0
                                                     )
    dense_model_accuracy = evaluate(
        model, dataloader['test'], criterion, device)
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(
        3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            logging.info(name)
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(
                sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'acc',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}%'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle(
        'Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    logging.info(f"=> Saving figure '{output}'")
    plt.savefig(output)


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log),
                                  logging.StreamHandler()
                                  ]
                        )
    logging.info("Arguments: " + str(args))

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = VGG('VGG19')
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

    # plot figures
    logging.info(f"=> Scaning weight_distribution")
    path = os.path.join(args.output, 'weight_distribution.png')
    plot_weight_distribution(net, path)

    logging.info(f"=> Scaning num_params_distribution")
    path = os.path.join(args.output, 'num_params_distribution.png')
    plot_num_parameters_distribution(net, path)

    logging.info(f"=> Scaning num_filters_distribution")
    path = os.path.join(args.output, 'num_filters_distribution.png')
    plot_num_filters_distribution(net, path)

    logging.info(f"=> Scaning weight_sensitivity_scan")
    path = os.path.join(args.output, 'weight_sensitivity_scan.png')
    plot_weight_sensitivity_scan(net, path)
