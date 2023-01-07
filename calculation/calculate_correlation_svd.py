import logging
import os
import argparse
import json
from tqdm.auto import tqdm
import torch

from similarity import similarity
from .decompose import get_num_unfold


def get_parser():
    parser = argparse.ArgumentParser(description='Precalculate distace/similarity through decomposition'
                                     )
    parser.add_argument('--correlation',
                        default=['cosine_sim', 'Pearson_sim',
                                 'Euclide_dis', 'Manhattan_dis', 'SNR_dis', 'VBD_dis'
                                 ],
                        type=str,
                        nargs='+',
                        help='correlation'
                        )
    parser.add_argument('--input',
                        default='calculation/baseline/vgg/decomposition_vgg16.json',
                        help='path to load calculated decomposition file',
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/correlation_decomposition_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_correlation_decomposition.txt',
                        help='path to log file',
                        )

    return parser


def get_correlation_mat(all_filters_u_dict, decomposer, criterion):
    """Caculate correlation matrix between channels in inter-layer
        based on precalculated left-singular vectors dict of hosvd
    """
    num_filters = len(all_filters_u_dict)
    # torch.zeros(num_filters, num_filters)
    correlation_mat = [[0.]*num_filters for i in range(num_filters)]
    num_unfold = get_num_unfold(decomposer)

    # compute the channel pairwise similarity based on criterion
    for i in tqdm(range(num_filters)):
        for j in range(num_filters):
            ui = all_filters_u_dict[str(i)]
            uj = all_filters_u_dict[str(j)]
            sum = 0.
            for x in range(num_unfold):
                sum += similarity(torch.tensor(ui[x]),
                                  torch.tensor(uj[x]),
                                  criterion
                                  )
            correlation_mat[i][j] = (sum/num_unfold).item()

    return correlation_mat


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log_file),
                                  logging.StreamHandler()
                                  ]
                        )
    logging.info("Arguments: " + str(args))

    # load pre-calculated decomposition
    logging.info(f"=> loading '{args.input}'")
    f = open(args.input, 'r+')
    data = json.loads(f.read())
    dict = {
        "input": args.input,
        "net": data["arch"],
        "checkpoint": data["checkpoint"]
    }

    # calculate correlation matrix through decomposition
    decomposition_dict = {}
    for decomposer, all_convs_dict in data["decomposition"].items():
        logging.info(f"------------------{decomposer}------------------")
        correlation_dict = {}
        for criterion in args.correlation:
            logging.info(f"----------------{criterion}----------------")
            all_convs_cor_mat = {}
            for i_conv, all_filters_u_dict in all_convs_dict.items():
                logging.info(f"------------i_conv: {i_conv}------------")
                correl_matrix = get_correlation_mat(
                    all_filters_u_dict, decomposer, criterion)
                all_convs_cor_mat[i_conv] = correl_matrix
            correlation_dict[criterion] = all_convs_cor_mat
        decomposition_dict[decomposer] = correlation_dict

    dict["correlation"] = decomposition_dict

    with open(args.output, 'w+') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)
