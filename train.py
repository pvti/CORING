'''Train CIFAR10'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import *
from torch.optim.lr_scheduler import *

import logging
import os
import argparse
import numpy as np
from tqdm.auto import tqdm

from data import *
from models import *
from utils import progress_bar


def get_parser():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--arch",
                        default="VGG16",
                        help="network architechture",
                        )
    parser.add_argument("--lr",
                        default=0.1,
                        type=float,
                        help="learning rate")
    parser.add_argument("--resume",
                        "-r",
                        action="store_true",
                        help="resume from checkpoint",
                        )
    parser.add_argument("--checkpoint",
                        default="checkpoint/baseline/vgg/vgg16.pth",
                        metavar="FILE",
                        help="path to load checkpoint",
                        )
    parser.add_argument("--output",
                        default="checkpoint/baseline/vgg/vgg16.pth",
                        help="path to save checkpoint",
                        )
    parser.add_argument("--num_epochs",
                        default=500,
                        type=int,
                        help="Number of training epochs",
                        )
    parser.add_argument("--log_file",
                        default="logs/log_train.txt",
                        help="path to log file",
                        )

    return parser


def train(net: nn.Module,
          dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: Optimizer,
          scheduler: LambdaLR,
          device: str,
          callbacks=None,
          verbose=True
          ) -> None:
    net.train()  # sets the mode to train
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if verbose:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(net: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: str,
             verbose=True,
             ) -> float:
    net.eval()  # sets the mode to evaluate

    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if verbose:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (correct / total * 100)


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
    best_acc = 0
    start_epoch = 0
    net = None
    if 'ResNet' in args.arch:
        net = getResNet(args.arch)
    elif 'VGG' in args.arch:
        net = VGG(args.arch)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        assert os.path.isfile(args.checkpoint), "Checkpoint not found!"
        logging.info(f"Loading {args.checkpoint} state ...")
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        logging.info(f"best_acc = {best_acc}, start_epoch = {start_epoch}")

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4,
                          )
    num_epochs = args.num_epochs
    steps_per_epoch = len(dataloader["train"])
    # Define the piecewise linear scheduler

    def lr_lambda(step): return np.interp([step / steps_per_epoch],
                                          [0, num_epochs * 0.3, num_epochs],
                                          [0, 1, 0]
                                          )[0]
    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        train(net,
              dataloader["train"],
              criterion,
              optimizer,
              scheduler,
              device
              )
        acc = evaluate(net,
                       dataloader["test"],
                       criterion,
                       device
                       )
        logging.info(f"epoch {epoch}:, {round(acc, 2)}")

        # save checkpoint if acc > best_acc
        if acc > best_acc:
            logging.info("Saving network state ...")
            state = {'net': net.state_dict(),
                     'acc': acc,
                     'epoch': epoch,
                     }
            torch.save(state, args.output)
            best_acc = acc
