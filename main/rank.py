from tqdm.auto import tqdm
import os
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname('..'))
from similarity import similarity
from ranking_strategy import get_saliency
from decompose import decompose

def get_correlation_mat(all_filters_u_dict, criterion='vbd_dis'):
    """Caculate correlation matrix between channels in inter-layer
        based on precalculated left-singular vectors dict of hosvd
    """
    num_filters = len(all_filters_u_dict)
    correlation_mat = [[0.]*num_filters for i in range(num_filters)]

    # compute the channel pairwise similarity based on criterion
    for i in tqdm(range(num_filters)):
        for j in range(num_filters):
            ui = all_filters_u_dict[i]
            uj = all_filters_u_dict[j]
            sum = 0.
            for x in range(3):
                sum += similarity(torch.tensor(ui[x]),
                                  torch.tensor(uj[x]),
                                  criterion=criterion
                                  )
            correlation_mat[i][j] = (sum/3).item()

    return correlation_mat


def get_rank(weight, criterion='vbd_dis', strategy='min_sum'):
    """Get rank based on HOSVD + criterion + strategy"""
    num_filters = weight.size(0)
    all_filters_u_dict = {}
    for i_filter in tqdm(range(num_filters)):
        filter = weight.detach()[i_filter, :]
        u = decompose(filter, decomposer='hosvd')
        all_filters_u_dict[i_filter] = [i.tolist() for i in u]

    correl_matrix = get_correlation_mat(all_filters_u_dict, criterion)
    correl_matrix = np.array(correl_matrix)

    dis = 1 if 'dis' in criterion else -1
    saliency = get_saliency(correl_matrix, strategy, dis)

    return saliency
