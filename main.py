from tqdm.auto import tqdm
import logging
import os
import argparse

from data import *
from helpers import *
from model import *
from pruner import channel_prune, apply_channel_sorting

def get_parser():
    parser = argparse.ArgumentParser(description="Filter pruning through SNR")
    parser.add_argument(
        "--input_baseline",
        default="model/baseline/vgg/vgg.cifar.pretrained.pth",
        metavar="FILE",
        help="path to load baseline model file",
    )
    parser.add_argument(
        "--output_pruned",
        default="model/pruned/vgg/",
        help="path to folder to save pruned model files",
    )
    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=0.2,
        help="Prune ratio",
    )
    parser.add_argument(
        "--num_finetune_epochs",
        type=int,
        default=5,
        help="Number of finetuning epochs",
    )
    parser.add_argument(
        "--log_file",
        default="./log.txt",
        help="path to log file",
    )

    return parser

def get_model_performance(model, dataloader):
    """Caculate model's performance
    """
    accuracy = round(evaluate(model, dataloader['test']), 2)
    size = round(get_model_size(model) / MiB, 2)

    #measure on cpu to simulate inference on an edge device
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')
    model = model.to('cpu')
    latency = round(measure_latency(model, dummy_input) * 1000, 1) #in ms
    macs = round(get_model_macs(model, dummy_input) / 1e6) #in million
    num_params = round(get_num_parameters(model)/ 1e6, 2)
    model = model.to('cuda')

    return accuracy, size, latency, macs, num_params

def finetune(epochs: int,
             model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             optimizer: Optimizer,
             scheduler: LambdaLR):
    """Finetune model, also return best accuracy
    """
    best_accuracy = 0
    for epoch in range(epochs):
        train(pruned_model, dataloader['train'], criterion, optimizer, scheduler)
        accuracy = evaluate(pruned_model, dataloader['test'])
        best_accuracy = max(best_accuracy, accuracy)
        #print(f'Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')

    return model, best_accuracy

if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.DEBUG)
    logging.info("Arguments: " + str(args))

    CIFAR10_dataset = DataLoaderCIFAR10()
    dataloader = CIFAR10_dataset.dataloader

    if args.input_baseline:
        #load pretrained model
        checkpoint = torch.load(args.input_baseline, map_location="cpu")
        model = VGG().cuda()
        logging.info(f"=> loading checkpoint '{args.input_baseline}'")
        model.load_state_dict(checkpoint['state_dict'])
        #recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])

        logging.info("Baseline model performance:")
        logging.info(get_model_performance(model, dataloader))

        channel_pruning_ratios = [0.1, 0.2, 0.3, 0.4]  # pruned-out ratio
        criterias = ['random', 'L0_norm', 'L1_norm', 'L2_norm', 'inf_norm','SNR', 'cos', 'EDistance']
        num_finetune_epochs = args.num_finetune_epochs
        for channel_pruning_ratio in channel_pruning_ratios:
            logging.info(f"-----Current channel_pruning_ratio-----{channel_pruning_ratio}")
            for criteria in criterias:
                sorted_model = apply_channel_sorting(model, criteria)
                pruned_model = channel_prune(sorted_model, channel_pruning_ratio)
                pruned_model_accuracy = evaluate(pruned_model, dataloader['test'])
                logging.info(f"{criteria}, accuracy = {pruned_model_accuracy:.2f}%")
                #finetune then save
                optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01,
                                        momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_finetune_epochs)
                criterion = nn.CrossEntropyLoss()
                finetuned_model, finetuned_model_best_acc = finetune(
                    num_finetune_epochs, pruned_model,
                    dataloader, criterion, optimizer, scheduler)
                logging.info(f"finetuned_model_best_acc = {finetuned_model_best_acc:.2f}%")
                path_save_model = os.path.join(
                args.output_pruned, f"{criteria}_{channel_pruning_ratio}.pth")
                torch.save(finetuned_model.state_dict(), path_save_model)
                logging.info(get_model_performance(finetuned_model, dataloader))
    else:
        logging.warning("Missing input baseline model")
        #train model in other process
