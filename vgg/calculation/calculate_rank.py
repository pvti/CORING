
from tqdm.auto import tqdm
import logging
import copy
import os
import argparse
import json
import numpy as np

from ranking_strategy import get_saliency


def get_parser():
    parser = argparse.ArgumentParser(description='Calculate saliency and rank based on calculated correlation'
                                     )
    parser.add_argument('--input',
                        default='calculation/baseline/vgg/correlation_vgg16.json',
                        help='path to load calculated correlation file',
                        )
    parser.add_argument('--strategy',
                        default=['sum', 'min_sum', 'min_min'],
                        type=str,
                        nargs='+',
                        help='filter ranking strategy'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/rank_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_rank_vgg16.txt',
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

    # load pre-calculated correlation
    logging.info(f"=> loading '{args.input}'")
    f = open(args.input, 'r+')
    data = json.loads(f.read())
    dict = {
        "input": args.input,
        "arch": data["arch"],
        "checkpoint": data["checkpoint"]
    }

    # calculate saliency and rank for correlation with strategy
    for strategy in args.strategy:
        logging.info(f"-------------------{strategy}-------------------")
        all_decomposer_dict = {}
        for decomposer, decomposition_dict in data["correlation"].items():
            logging.info(f"----------------{decomposer}----------------")
            saliency_dict = {}
            sort_idx_dict = {}
            for criterion, all_convs_cor_mat in decomposition_dict.items():
                logging.info(f"----------------{criterion}----------------")
                all_convs_saliency = {}
                all_convs_sort_idx = {}
                dis = 1 if 'dis' in criterion else -1
                for i_conv, correl_matrix in tqdm(all_convs_cor_mat.items()):
                    mat = np.array(correl_matrix)
                    saliency = get_saliency(mat, strategy, dis)
                    sort_idx = saliency.argsort(kind='stable')[
                        ::-1][:len(saliency)]
                    all_convs_saliency[i_conv] = saliency.tolist()
                    all_convs_sort_idx[i_conv] = sort_idx.tolist()
                saliency_dict[criterion] = all_convs_saliency
                sort_idx_dict[criterion] = all_convs_sort_idx
            all_decomposer_dict[decomposer] = {
                "saliency": saliency_dict,
                "sort_idx": sort_idx_dict
                }
        dict[strategy] = all_decomposer_dict

    with open(args.output, 'w+') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)
