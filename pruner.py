import copy
import torch
from torch import nn
from typing import Union, List
import numpy as np
from similarity import *


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """

    return int(round(channels * (1. - prune_ratio)))


@torch.no_grad()
def channel_prune(model: nn.Module,
                  prune_ratio: Union[List, float]) -> nn.Module:
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv - 1)

    # we prune the convs in the backbone with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite
    # we only apply pruning to the backbone features
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    assert len(all_convs) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_convs[i_ratio]
        prev_bn = all_bns[i_ratio]
        next_conv = all_convs[i_ratio + 1]
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

        # prune the input of the next conv (hint: just one line of code)
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])

    return model


def get_input_channel_similarity(weight, criteria):
    """Caculate correlation matrix between channels in inter-layer
    """
    in_channels = weight.shape[1]
    correlation_matrix = torch.zeros(in_channels, in_channels,
                                     device=torch.device('cuda'))

    # compute the channel pairwise similarity based on criteria
    if criteria == 'cosine_sim':
        for ic in range(in_channels-1):
            for jc in range(ic+1, in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = cosine(channel_weight_ic,
                                                    channel_weight_jc)
                correlation_matrix[jc, ic] = correlation_matrix[ic, jc]

    elif criteria == 'Euclide_dis':
        for ic in range(in_channels-1):
            for jc in range(ic+1, in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = Euclide(channel_weight_ic,
                                                     channel_weight_jc)
                correlation_matrix[jc, ic] = correlation_matrix[ic, jc]

    elif criteria == 'Manhattan_dis':
        for ic in range(in_channels-1):
            for jc in range(ic+1, in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = Manhattan(channel_weight_ic,
                                                       channel_weight_jc)
                correlation_matrix[jc, ic] = correlation_matrix[ic, jc]

    elif criteria == 'Pearson_sim':
        for ic in range(in_channels-1):
            for jc in range(ic+1, in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = Pearson(channel_weight_ic,
                                                     channel_weight_jc)
                correlation_matrix[jc, ic] = correlation_matrix[ic, jc]

    elif criteria == 'SNR_dis':
        for ic in range(in_channels):
            for jc in range(in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = SNR(channel_weight_ic,
                                                 channel_weight_jc)

    return correlation_matrix


def get_input_channel_saliency_by_norm(weight, criteria):
    in_channels = weight.shape[1]
    saliencies = []

    # compute the importance for each input channel
    if (criteria == 'L0_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(
                torch.flatten(channel_weight), 0)
            saliencies.append(saliency.view(1))

    elif (criteria == 'L1_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(
                torch.flatten(channel_weight), 1)
            saliencies.append(saliency.view(1))

    elif (criteria == 'L2_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(channel_weight)
            saliencies.append(saliency.view(1))

    elif (criteria == 'inf_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(torch.flatten(channel_weight),
                                                float('inf'))
            saliencies.append(saliency.view(1))

    return saliencies


def get_input_channel_saliency(weight, criteria):
    """Caculate saliencies of channels in inter-layer
    """
    saliencies = []
    if (criteria == 'random'):
        saliencies = torch.from_numpy(np.random.uniform(
            low=0, high=weight.shape[1], size=weight.shape[1])
            .astype('uint8')).to(0)

        return saliencies

    elif ('norm' in criteria):
        saliencies = get_input_channel_saliency_by_norm(weight, criteria)

        return torch.cat(saliencies)

    elif ('sim' in criteria):
        correlation_matrix = get_input_channel_similarity(weight, criteria)
        #return get_saliency(correlation_matrix, dis = -1)
        for ic in range(weight.shape[1]):
            saliency = - correlation_matrix[ic, :].sum()
            saliencies.append(saliency.view(1))

    elif ('dis' in criteria):
        correlation_matrix = get_input_channel_similarity(weight, criteria)
        #return get_saliency(correlation_matrix, dis = 1)
        for ic in range(weight.shape[1]):
            saliency = correlation_matrix[ic, :].sum()
            saliencies.append(saliency.view(1))

    return torch.cat(saliencies)


def compare_min(row, col, matrix, dis = 1):
    """Compare minimum distance/similarity from row, col to other elements
    Matrix must be symmetrix
    """
    if dis == 1:
        min_row = np.min(matrix[row, :])
        min_col = np.min(matrix[:, col])

        return col if min_row > min_col else row

    else:
        max_row = np.max(matrix[row, :])
        max_col = np.max(matrix[:, col])

        return col if max_row < max_col else row


def compare_sum(row, col, matrix, inf, dis = 1):
    """Compare sum of distance/similarity from row, col to other elements
    Matrix must be symmetrix
    """
    num_row = matrix.shape[0]
    if dis == 1:
        min_row = min_col = 0
        for i in range(num_row):
            if matrix[row, i] != inf:
                min_row += matrix[row, i]
            if matrix[i, col] != inf:
                min_col += matrix[i, col]

        return col if min_row > min_col else row

    else:
        max_row = max_col = 0
        for i in range(num_row):
            if matrix[row, i] != inf:
                max_row += matrix[row, i]
            if matrix[i, col] != inf:
                max_col += matrix[i, col]

        return col if max_row < max_col else row


def get_saliency(correlation, dis = 1):
    """Sort saliency based on distance/similarity matrix
    """
    mat = copy.deepcopy(correlation.detach().cpu().numpy())
    num_row = mat.shape[0]
    inf = dis*float('inf')
    for i in range(num_row):
        mat[i, i]  = inf

    saliency = torch.full((num_row,), num_row-1).cuda()

    for i in range(num_row-1):
        if dis ==1:
            row, col = np.unravel_index(mat.argmin(), mat.shape)
        else:
            row, col = np.unravel_index(mat.argmax(), mat.shape)
        mat[row, col] = inf
        mat[col, row] = inf
        index = compare_min(row, col, mat, dis)
        #index = compare_sum(row, col, mat, inf, dis)
        mat[index, :] = inf
        mat[:, index] = inf
        saliency[index] = i

    return saliency

@torch.no_grad()
def apply_channel_sorting(model, criteria):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_saliency(next_conv.weight, criteria)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(
            prev_conv.weight.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input
        next_conv.weight.copy_(
            torch.index_select(next_conv.weight.detach(), 1, sort_idx))

    return model
