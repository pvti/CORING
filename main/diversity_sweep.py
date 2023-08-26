import argparse
import wandb
import torch
import torch.nn as nn
import numpy as np
from models.cifar10.vgg import vgg_16_bn
from rank import get_rank
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("VGG-16 diversity measure")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/cifar10/vgg_16_bn.pt",
        help="checkpoint model path",
    )
    parser.add_argument(
        "--decomposer",
        type=str,
        default="hosvd",
        choices=("svd", "hosvd", "tucker"),
        help="decomposition method",
    )
    parser.add_argument(
        "--criterion",
        default="Euclide_dis",
        type=str,
        choices=("Euclide_dis", "VBD_dis", "cosine_sim"),
        help="criterion",
    )
    parser.add_argument("--strategy", default="min_sum", type=str, help="strategy")
    parser.add_argument(
        "-kr", "--keep-ratio", type=float, default=0.75, help="keep ratio"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="unfolding mode, for svd only",
    )
    parser.add_argument(
        "--rank", type=int, default=1, choices=(1, 2, 3), help="decomposition rank"
    )

    return parser.parse_args()


def pairwise_euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2)


def calculate_euclidean_distance(filters):
    total_euclidean_distance = 0.0
    num_pairs = 0

    for i in range(len(filters)):
        for j in range(i + 1, len(filters)):
            distance = pairwise_euclidean_distance(filters[i], filters[j])
            total_euclidean_distance += distance
            num_pairs += 1

    average_euclidean_distance = total_euclidean_distance / num_pairs

    return total_euclidean_distance, average_euclidean_distance


def calculate_entropy(tensor):
    probabilities = np.abs(tensor.cpu().numpy())  # Use absolute values for simplicity
    probabilities /= np.sum(probabilities)  # Normalize to make them probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def svd_complexity(M, N, R=1):
    """
    Compute the complexity of the SVD of a matrix of size MxN for rank R
    """

    return 2 * R * M * N + 2 * R**2 * (M + N)


def compute_params_complexity(filters, decomposer="hosvd", rank=1, mode=0):
    """
    Compute the number of parameters of the representation and the complexity of the decomposition process
    """
    out_channels, in_channels, kernel_size, _ = filters.size()
    # 1 filter
    if decomposer == "svd":
        if mode == 0:  # Cin x (dxd)
            params = rank * (in_channels + kernel_size**2)
            complexity = svd_complexity(in_channels, kernel_size**2, rank)
        elif mode == 1 or mode == 2:  # (dxCin) x d
            params = rank * (in_channels * kernel_size + kernel_size)
            complexity = svd_complexity(in_channels * kernel_size, kernel_size, rank)
    else:
        params = rank * (in_channels + 2 * kernel_size)
        complexity = svd_complexity(
            in_channels, kernel_size**2, rank
        ) + 2 * svd_complexity(kernel_size, in_channels * kernel_size, rank)
    # all filters
    params *= out_channels
    complexity *= out_channels

    return params, complexity


def process_1_layer(
    layer: nn.Conv2d,
    keep_ratio: float,
    decomposer="hosvd",
    rank=1,
    mode=0,
    criterion="Euclide_dis",
    strategy="min_sum",
):
    weight = layer.weight.data
    number_filters_ori = weight.size(0)
    number_filters_selected = int(number_filters_ori * keep_ratio)
    saliency = get_rank(
        weight=weight,
        decomposer=decomposer,
        rank=rank,
        mode=mode,
        criterion=criterion,
        strategy=strategy,
    )
    select_index = np.argsort(saliency)[
        number_filters_ori - number_filters_selected :
    ]  # preserved filter id
    select_index.sort()
    filters_selected = weight[select_index]

    total_euclidean_distance, average_euclidean_distance = calculate_euclidean_distance(
        filters_selected
    )

    # entropy
    reshaped_filters = filters_selected.view(number_filters_selected, -1)
    filter_entropies = [calculate_entropy(filter) for filter in reshaped_filters]
    average_entropy = np.mean(filter_entropies)

    params, complexity = compute_params_complexity(
        filters_selected,
        decomposer=decomposer,
        rank=rank,
        mode=mode,
    )

    return (
        total_euclidean_distance,
        average_euclidean_distance,
        average_entropy,
        params,
        complexity,
    )


def main():
    args = parse_args()
    project_name = f"CORING DiversitySweep rank={args.rank} criterion={args.criterion} strategy={args.strategy}"
    wandb.init(
        name=f"sweeper",
        project=project_name,
        config=vars(args),
    )
    # load checkpoint
    model = vgg_16_bn(compress_rate=[0.0] * 14).cuda()
    checkpoint = torch.load(args.ckpt, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])

    for name, module in model.named_modules():
        name = name.replace("module.", "")
        if isinstance(module, nn.Conv2d):
            print(f"processing {name}")
            max_diff = 0
            best_kr = 0.25
            for kr in np.arange(0.25, 0.8, 0.05):
                _, tensor_avg, _, _, _ = process_1_layer(
                    layer=module, keep_ratio=kr, decomposer="hosvd", rank=1, mode=0
                )
                _, matrix_avg, _, _, _ = process_1_layer(
                    layer=module, keep_ratio=kr, decomposer="svd", rank=1, mode=0
                )

                diff = tensor_avg / matrix_avg
                if diff > max_diff:
                    max_diff = diff
                    best_kr = kr

            layer_th = int(name[13:])  # features.conv0 => 0
            wandb.log(
                {
                    "layer_th": layer_th,
                    "max_diff": max_diff,
                    "best_kr": best_kr,
                }
            )


if __name__ == "__main__":
    main()
