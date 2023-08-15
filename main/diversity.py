import argparse
import wandb
import torch
import torch.nn as nn
import numpy as np
from models.cifar10.vgg import vgg_16_bn
from rank import get_rank


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


def compute_params_complexity(filters, decomposer="hosvd"):
    """
    Compute the number of parameters of the representation and the complexity of the decomposition process
    """
    out_channels, in_channels, kernel_size, _ = filters.size()
    # 1 filter
    if decomposer == "svd":
        params = in_channels + kernel_size**2
        complexity = svd_complexity(in_channels, kernel_size**2)
    else:
        params = in_channels + 2 * kernel_size
        complexity = svd_complexity(in_channels, kernel_size**2) + 2 * svd_complexity(
            kernel_size, in_channels * kernel_size
        )
    # all filters
    params *= out_channels
    complexity *= out_channels

    return params, complexity


def process_1_layer(
    layer: nn.Conv2d,
    keep_ratio: float,
    decomposer="hosvd",
    criterion="Euclide_dis",
    strategy="min_sum",
):
    weight = layer.weight.data
    number_filters_ori = weight.size(0)
    number_filters_selected = int(number_filters_ori * keep_ratio)
    saliency = get_rank(
        weight=weight,
        decomposer=decomposer,
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
        filters_selected, decomposer=decomposer
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
    project_name = f"CORING DiversityComparison criterion={args.criterion} strategy={args.strategy}"
    wandb.init(
        name=f"{args.decomposer} {args.keep_ratio}",
        project=project_name,
        config=vars(args),
    )

    # load checkpoint
    model = vgg_16_bn(compress_rate=[0.0] * 14).cuda()
    checkpoint = torch.load(args.ckpt, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])
    # module = model.features._modules[args.layer]

    for name, module in model.named_modules():
        name = name.replace("module.", "")
        if isinstance(module, nn.Conv2d):
            print(f"processing {name}")
            (
                Eu_distance_sum,
                Eu_distance_avg,
                entropy,
                params,
                complexity,
            ) = process_1_layer(
                layer=module,
                keep_ratio=args.keep_ratio,
                decomposer=args.decomposer,
                criterion=args.criterion,
                strategy=args.strategy,
            )
            layer_th = int(name[13:])  # features.conv0 => 0
            wandb.log(
                {
                    "layer_th": layer_th,
                    "Euclide_distance_sum": Eu_distance_sum,
                    "params": params,
                    "complexity": complexity,
                }
            )


if __name__ == "__main__":
    main()
