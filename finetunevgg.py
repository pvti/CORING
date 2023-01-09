import argparse
from tqdm.auto import tqdm
import logging
import os
import wandb
import torch

from data import DataLoaderCIFAR10
from helpers import get_model_performance
from models import VGG
from train import train, evaluate


def get_parser():
    """Define the command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description='Finetune pruned VGG16')
    parser.add_argument('--checkpoint',
                        default='checkpoint/pruned/vgg16/vgg16_hosvd_VBD_dis_min_sum_Acc.pth', metavar='FILE',
                        help='path to load pruned checkpoint')
    parser.add_argument('--output', default='checkpoint/finetuned/vgg16/',
                        help='path to save finetuned checkpoint')
    parser.add_argument('--log', default='logs/',
                        help='path to log folder')
    parser.add_argument('--device', default=0, type=int,
                        help='device to use for training')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    return parser


def main(args):
    """Main function for the script."""
    # Process output file path
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    _, tail = os.path.split(args.checkpoint)
    name = tail.split('.')[0]  # remove '.pth'
    log_pth = os.path.join(args.log, name+'.txt')
    out_pth = os.path.join(args.output, tail)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_pth), logging.StreamHandler()])
    logging.info("Arguments: " + str(args))

    # Load the CIFAR10 dataset and define the loss function
    cifar10_dataset = DataLoaderCIFAR10()
    dataloader = cifar10_dataset.dataloader
    criterion = torch.nn.CrossEntropyLoss()

    # Init wandb name
    wandb.init(name=name, project='DVBD', config={**vars(args)})

    # Set the device to use for training
    device = 'cuda:{}'.format(
        args.device) if torch.cuda.is_available() else 'cpu'

    # Load the state dictionary for the pre-trained model from the checkpoint file
    assert os.path.isfile(args.checkpoint), "Checkpoint not found!"
    logging.info(f"=> Loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint)
    net = VGG('VGG16', cfg=checkpoint['cfg'])
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    logging.info(f'Pruned model: best_acc (%) = {best_acc} size (Mb), latency (ms), macs (M), num_params (M) = '
                 f'{get_model_performance(net)}')
    logging.info(net)

    # Finetune model
    logging.info(f"=> Finetuning model...")
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)
    for epoch in tqdm(range(start_epoch, start_epoch+args.epochs)):
        train(net, dataloader['train'], criterion,
              optimizer, scheduler, device)
        acc = evaluate(net, dataloader['test'], criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        wandb.log({'acc': acc, 'best_acc': best_acc, 'lr': cur_lr})
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch
            }
            torch.save(state, out_pth)
            best_acc = acc


if __name__ == "__main__":
    main(get_parser().parse_args())
