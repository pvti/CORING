from tqdm.auto import tqdm
import os
import torch
import numpy as np

import sys

sys.path.append(os.path.dirname(".."))
from similarity import similarity
from ranking_strategy import get_saliency
from decompose import decompose, get_num_unfold


def get_correlation_mat(all_filters_u_dict, criterion="VBD_dis"):
    """Caculate correlation matrix between channels in inter-layer
    based on precalculated left-singular vectors dict of hosvd
    """
    num_filters = len(all_filters_u_dict)
    correlation_mat = [[0.0] * num_filters for i in range(num_filters)]

    # compute the channel pairwise similarity based on criterion
    for i in tqdm(range(num_filters)):
        for j in range(num_filters):
            ui = all_filters_u_dict[i]
            uj = all_filters_u_dict[j]
            fold = len(ui)
            sum = 0.0
            for x in range(fold):
                sum += similarity(
                    torch.tensor(ui[x]), torch.tensor(uj[x]), criterion=criterion
                )
            correlation_mat[i][j] = (sum / fold).item()

    return correlation_mat


def get_rank(
    weight, decomposer="hosvd", rank=1, mode=0, criterion="VBD_dis", strategy="min_sum"
):
    """Get saliency based on decomposer + rank + criterion + strategy"""
    num_filters = weight.size(0)
    all_filters_u_dict = {}
    for i_filter in tqdm(range(num_filters)):
        filter = weight.detach()[i_filter, :]
        u = decompose(filter, decomposer=decomposer, rank=rank, mode=mode)
        all_filters_u_dict[i_filter] = [i.tolist() for i in u]

    correl_matrix = get_correlation_mat(all_filters_u_dict, criterion)
    correl_matrix = np.array(correl_matrix)

    dis = 1 if "dis" in criterion else -1
    saliency = get_saliency(correl_matrix, strategy, dis)

    return saliency
