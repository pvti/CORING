from tqdm.auto import tqdm
import logging
import os
import argparse
import numpy as np
from collections import OrderedDict
import json

from data import *
from helpers import get_model_performance
from models import *
from pruner import channel_prune, apply_channel_sorting
from train import train, evaluate
from miscellaneous.plot import plot_decomposer


def get_parser():
    parser = argparse.ArgumentParser(description='Similarity based filter pruning'
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
    parser.add_argument('--decomposer',
                        default=['full', 'svd', 'hosvd'],
                        type=str,
                        nargs='+',
                        help='tensor decomposer'
                        )
    parser.add_argument('--calculated_rank',
                        default='calculation/baseline/vgg/rank_vgg16.json',
                        help='path to load pre-calculated rank file',
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
                        default='logs/log_main_vgg16.txt',
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

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

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
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    logging.info(
        f'Baseline model: best_acc (%) = {best_acc} size (Mb), latency (ms), macs (M), num_params (M) = {get_model_performance(net)}')

    prune_ratios = np.arange(args.prune_range[0],
                             args.prune_range[1],
                             args.prune_range[2]
                             )
    prune_ratios = np.around(prune_ratios, 2)

    num_finetune_epochs = args.num_finetune_epochs

    # load pre-calculated sort idx
    logging.info(f"=> loading '{args.calculated_rank}'")
    f = open(args.calculated_rank, 'r')
    data = json.loads(f.read())

    pruned_acc_strategy = {s: {} for s in args.strategy}
    finetuned_acc_strategy = {s: {} for s in args.strategy}
    for strategy in args.strategy:
        logging.info(f"------------------{strategy}------------------")
        for decomposer in args.decomposer:
            logging.info(f"------------------{decomposer}------------------")
            pruned_accuracy_dict = {c: [] for c in args.criteria}
            ft_best_acc_dict = {c: [] for c in args.criteria}
            for criteria in args.criteria:
                logging.info(f"---------------{criteria}---------------")
                sort_idx_dict = data[strategy][decomposer]['sort_idx'][criteria]
                sorted_net = apply_channel_sorting(net, sort_idx_dict)
                for prune_ratio in tqdm(prune_ratios):
                    pruned_net = channel_prune(sorted_net, prune_ratio)
                    pruned_net_accuracy = evaluate(pruned_net,
                                                   dataloader['test'],
                                                   criterion,
                                                   device,
                                                   verbose=False
                                                   )

                    logging.info(
                        f"prune_ratio {prune_ratio}, pruned_net_acc = {pruned_net_accuracy:.2f}%")
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

                    ft_best_acc = 0
                    for epoch in range(start_epoch, start_epoch + num_finetune_epochs):
                        logging.info(
                            f"---------------finetuning epoch = {epoch}---------------")
                        train(pruned_net, dataloader['train'],
                              criterion, optimizer, scheduler, device)
                        acc = evaluate(pruned_net, dataloader['test'],
                                       criterion, device)

                        # save checkpoint if acc > best_acc
                        if acc > ft_best_acc:
                            '''
                            state = {'net': pruned_net.state_dict(),
                                     'acc': acc,
                                     'epoch': epoch,
                                     }
                            path_save_net = os.path.join(args.output,
                                                         f"{criteria}_{prune_ratio}.pth")
                            torch.save(state, path_save_net)
                            '''
                            ft_best_acc = acc

                    ft_best_acc_dict[criteria].append(round((ft_best_acc), 2))

                    if int(100*prune_ratio) % 10 == 0:
                        logging.info(
                            f'Finetune pruned model: ft_best_acc (%) = {ft_best_acc} size (Mb), latency (ms), macs (M), num_params (M) = {get_model_performance(pruned_net)}')

            pruned_acc_strategy[strategy][decomposer] = pruned_accuracy_dict
            finetuned_acc_strategy[strategy][decomposer] = ft_best_acc_dict

    logging.info(pruned_acc_strategy)
    logging.info(finetuned_acc_strategy)

    # Plot decomposers comparison
    for strategy, strategy_dict in pruned_acc_strategy.items():
        path = os.path.join('figures/decomposer/pruned/', strategy + '.png')
        logging.info(f"=> Saving figure '{path}'")
        plot_decomposer(strategy_dict, prune_ratios, path)
    for strategy, strategy_dict in finetuned_acc_strategy.items():
        path = os.path.join('figures/decomposer/finetuned/', strategy + '.png')
        logging.info(f"=> Saving figure '{path}'")
        plot_decomposer(strategy_dict, prune_ratios, path)
