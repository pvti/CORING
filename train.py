import logging
import os
import argparse
import numpy as np

from data import *
from model import *

def get_parser():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        help="path to load baseline model file",
    )
    parser.add_argument(
        "--output",
        default="model/baseline/vgg/model.pth",
        help="path to save model file",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--log_file",
        default="./log_train.txt",
        help="path to log file",
    )

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(filename=args.log_file,
                        encoding='utf-8', level=logging.DEBUG)
    logging.info("Arguments: " + str(args))

    model = VGG().cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4,)
    num_epochs = args.num_epochs
    steps_per_epoch = len(dataloader["train"])
    # Define the piecewise linear scheduler
    lr_lambda = lambda step: np.interp([step / steps_per_epoch],
                                       [0, num_epochs * 0.3, num_epochs],
                                       [0, 1, 0])[0]
    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch_num in tqdm(range(1, num_epochs + 1)):
        train(model, dataloader["train"], criterion, optimizer, scheduler)
        metric = evaluate(model, dataloader["test"])
        logging.info(f"epoch {epoch_num}:, {metric}")

    torch.save(model.state_dict(), args.output)
