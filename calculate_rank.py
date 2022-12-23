
from tqdm.auto import tqdm
import logging
import copy
import os
import argparse
import json
import numpy as np
from ranking_strategy import get_saliency


def get_parser():
    parser = argparse.ArgumentParser(description='Calculate saliency and rank based on calculated correlation/norm'
                                     )
    parser.add_argument('--input',
                        default='calculation/baseline/vgg/criteria.json',
                        help='path to load calculated correlation/norm file',
                        )
    parser.add_argument('--strategy',
                        default=['sum', 'min_sum', 'min_min'],
                        type=str,
                        nargs='+',
                        help='filter ranking strategy'
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/rank.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/calculation_rank.txt',
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

    with open(args.input, 'r+') as file:
        logging.info(f"=> loading '{args.input}'")
        data = json.load(file)
        dict = {
            "net": data["net"],
            "checkpoint": data["checkpoint"]
        }

        # calculate rank for norm
        sort_idx_dict = {}
        for norm, all_convs_norm in data["norm"].items():
            logging.info(f"-----{norm}-----")
            all_convs_sort_idx = {}
            for i_conv, norm_arr in tqdm(all_convs_norm.items()):
                arr = np.array(norm_arr)
                # sort descending = True
                sort_idx = arr.argsort(kind='stable')[::-1][:len(arr)]
                all_convs_sort_idx[i_conv] = sort_idx.tolist()
            sort_idx_dict[norm] = all_convs_sort_idx

        # calculate saliency and rank for correlation with strategy
        for strategy in args.strategy:
            logging.info(f"---------------{strategy}---------------")
            saliency_dict = {}
            for criterion, all_convs_cor_mat in data["correlation"].items():
                logging.info(f"-----{criterion}-----")
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

            dict[strategy] = {
                "saliency": saliency_dict,
                # sort_idx_dict CHANGED in loop
                "sort_idx": copy.deepcopy(sort_idx_dict)
            }

        with open(args.output, 'w+') as file:
            logging.info(f"=> dumping to {args.output}")
            json.dump(dict, file, indent=4)
