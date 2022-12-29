from tqdm.auto import tqdm
import logging
import os
import argparse
import numpy as np
from collections import OrderedDict
import json
import torch

from data import *
from helpers import *
from models import *
from pruner import channel_prune, apply_channel_sorting
from train import train, evaluate


def get_parser():
    parser = argparse.ArgumentParser(description='Norm-based filter pruning'
                                     )
    parser.add_argument('--arch',
                        default='VGG16',
                        help='network architecture',
                        )
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='learning rate for finetuning')
    parser.add_argument('--checkpoint',
                        default='checkpoint/baseline/vgg/vgg16.pth',
                        metavar='FILE',
                        help='path to load checkpoint',
                        )
    parser.add_argument('--criteria',
                        default=['L0', 'L1', 'L2', 'inf'],
                        type=str,
                        nargs='+',
                        help='norm'
                        )
    parser.add_argument('--calculated_norm',
                        default='calculation/baseline/vgg/norm_vgg16.json',
                        help='path to load pre-calculated norm file',
                        )
    parser.add_argument('--output',
                        default='checkpoint/pruned/vgg/',
                        help='path to save checkpoint',
                        )
    parser.add_argument('-p', '--prune_range',
                        default=(0.01, 0.501, 0.01),
                        type=float,
                        nargs=3,
                        metavar=('start', 'end', 'step'),
                        help='prune ratio range [start end step]'
                        )
    parser.add_argument('--num_finetune_epochs',
                        default=10,
                        type=int,
                        help='Number of finetuning epochs',
                        )
    parser.add_argument('--log_file',
                        default='logs/log_main_norm.txt',
                        help='path to log file',
                        )

    return parser


def get_model_performance(model, dataloader, criterion):
    """Caculate model's performance
    """
    accuracy = round(evaluate(
        model, dataloader['test'], criterion, device='cuda', verbose=False), 2)
    size = round(get_model_size(model) / MiB, 2)

    # measure on cpu to simulate inference on an edge device
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')
    model = model.to('cpu')
    latency = round(measure_latency(model, dummy_input) * 1000, 1)  # in ms
    macs = round(get_model_macs(model, dummy_input) / 1e6)  # in million
    num_params = round(get_num_parameters(model) / 1e6, 2)  # in million
    model = model.to('cuda')

    return accuracy, size, latency, macs, num_params


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
    start_epoch = 0

    net = None
    if args.arch == "ResNet18":
        net = ResNet18()
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
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    logging.info("Baseline model performance: \n" +
                 "accuracy (%), size (Mb), latency (ms), macs (M), num_params (M)"
                 )
    logging.info(get_model_performance(net, dataloader, criterion))

    channel_pruning_ratios = np.arange(args.prune_range[0],
                                       args.prune_range[1],
                                       args.prune_range[2]
                                       )
    channel_pruning_ratios = np.around(channel_pruning_ratios, 2)

    num_finetune_epochs = args.num_finetune_epochs

    # load pre-calculated sort idx
    logging.info(f"=> loading '{args.calculated_norm}'")
    f = open(args.calculated_norm, 'r')
    data = json.loads(f.read())

    pruned_acc_strategy = {}
    finetuned_acc_strategy = {}
    pruned_accuracy_dict = {c: [] for c in args.criteria}
    finetuned_best_acc_dict = {c: [] for c in args.criteria}
    for criteria in args.criteria:
        logging.info(f"---------------{criteria}---------------")
        sort_idx_dict = data['sort_idx'][criteria]
        sorted_net = apply_channel_sorting(net, sort_idx_dict)
        for channel_pruning_ratio in tqdm(channel_pruning_ratios):
            pruned_net = channel_prune(sorted_net, channel_pruning_ratio)
            pruned_net_accuracy = evaluate(pruned_net,
                                           dataloader['test'],
                                           criterion,
                                           device,
                                           verbose=False
                                           )

            logging.info(
                f"channel_pruning_ratio {channel_pruning_ratio}, pruned_net_acc = {pruned_net_accuracy:.2f}%")
            pruned_accuracy_dict[criteria].append(
                round((pruned_net_accuracy), 2))

            # finetune then save
            optimizer = torch.optim.SGD(pruned_net.parameters(),
                                        lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=1e-4
                                        )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   args.num_finetune_epochs)

            finetuned_best_accuracy = 0
            for epoch in range(start_epoch, start_epoch + num_finetune_epochs):
                logging.info(
                    f"---------------finetuning epoch = {epoch}---------------")
                train(pruned_net, dataloader['train'],
                      criterion, optimizer, scheduler, device)
                acc = evaluate(pruned_net, dataloader['test'],
                               criterion, device, verbose=False)

                # save checkpoint if acc > best_acc
                if acc > finetuned_best_accuracy:
                    state = {'net': net.state_dict(),
                             'acc': acc,
                             'epoch': epoch,
                             }
                    path_save_net = os.path.join(args.output,
                                                 f"{criteria}_{channel_pruning_ratio}.pth")
                    torch.save(state, path_save_net)
                    finetuned_best_accuracy = acc

            finetuned_best_acc_dict[criteria].append(
                round((finetuned_best_accuracy), 2))

            if int(100*channel_pruning_ratio) % 10 == 0:
                logging.info(get_model_performance(
                    pruned_net, dataloader, criterion))

    logging.info(pruned_accuracy_dict)
    logging.info(finetuned_best_acc_dict)
