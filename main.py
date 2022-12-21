from tqdm.auto import tqdm
import logging
import os
import argparse
import numpy as np
from collections import OrderedDict
import json

from data import *
from helpers import *
from models import *
from pruner import channel_prune, apply_channel_sorting
from train import train, evaluate


def get_parser():
    parser = argparse.ArgumentParser(description='Filter pruning through SNR'
                                     )
    parser.add_argument('--net',
                        default='VGG19',
                        help='network type',
                        )
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='learning rate for finetuning')
    parser.add_argument('-r', '--resume',
                        action='store_true',
                        help='resume from checkpoint',
                        )
    parser.add_argument('--checkpoint',
                        default='checkpoint/baseline/vgg/ckpt.pth',
                        metavar='FILE',
                        help='path to load checkpoint',
                        )
    parser.add_argument('--criteria',
                        default=['L0_norm', 'L1_norm', 'L2_norm', 'inf_norm',
                                 'cosine_sim', 'Pearson_sim', 'svd_sim', 'hosvd_sim',
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
    parser.add_argument('--calculated_rank',
                        default='calculation/baseline/vgg/rank.json',
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
                        default='logs/log_main.txt',
                        help='path to log file',
                        )

    return parser


def get_model_performance(model, dataloader, criterion):
    """Caculate model's performance
    """
    accuracy = round(evaluate(model, dataloader['test'], criterion, 'cuda'), 2)
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
    if (args.net == "ResNet18"):
        net = ResNet18()
    elif 'VGG' in args.net:
        net = VGG(args.net)
    else:
        net = VGG('VGG19')
    net = net.to(device)

    if args.resume:
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
    logging.info(f"=> loading '{args.calculated_rank}'")
    f = open(args.calculated_rank, 'r')
    data = json.loads(f.read())

    pruned_acc_strategy = {}
    finetuned_acc_strategy = {}
    for strategy in args.strategy:
        logging.info(f"------------------{strategy}------------------")
        pruned_accuracy_dict = {c: [] for c in args.criteria}
        finetuned_best_acc_dict = {c: [] for c in args.criteria}
        for criteria in args.criteria:
            logging.info(f"---------------{criteria}---------------")
            sort_idx_dict = data[strategy]['sort_idx'][criteria]
            sorted_net = apply_channel_sorting(net, sort_idx_dict)
            for channel_pruning_ratio in tqdm(channel_pruning_ratios):
                pruned_net = channel_prune(sorted_net, channel_pruning_ratio)
                pruned_net_accuracy = evaluate(pruned_net,
                                            dataloader['test'],
                                            criterion,
                                            device
                                            )

                logging.info(f"pruned_net_acc = {pruned_net_accuracy:.2f}%")
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
                    train(pruned_net, dataloader['train'],
                        criterion, optimizer, scheduler, device)
                    acc = evaluate(pruned_net, dataloader['test'],
                                criterion, device)

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

                logging.info(get_model_performance(
                    pruned_net, dataloader, criterion))

        pruned_acc_strategy[strategy] = pruned_accuracy_dict
        finetuned_acc_strategy[strategy] = finetuned_best_acc_dict
    #logging.info(pruned_accuracy_dict)
    #logging.info(finetuned_best_acc_dict)
    logging.info(pruned_acc_strategy)
    logging.info(finetuned_acc_strategy)
