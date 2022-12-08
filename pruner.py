import copy
import torch
from torch import nn
from typing import Union, List
from torchmetrics import SignalNoiseRatio
import numpy as np

snr = SignalNoiseRatio().to(0)
cos = torch.nn.CosineSimilarity(dim=0)

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

def signal_to_noise_ratio(signal, noise):
    """Caculate signal to noise ratio
    """

    return snr(signal, noise)

def EuclideDistance(a, b):
    """Caculate Euclide distance
    """

    return torch.dist(a, b)

def cosineSimilarity(a, b):
    """Caculate cosine similarity
    """

    return cos(torch.flatten(a), torch.flatten(b))

def get_input_channel_similarity(weight, criteria):
    """Caculate correlation matrix between channels in inter-layer
    """
    in_channels = weight.shape[1]
    correlation_matrix = torch.zeros(in_channels, in_channels,
                                     device=torch.device('cuda'))

    # compute the similarity for each channel pair based on criteria
    if criteria == 'SNR':
      for ic in range(in_channels):
        for jc in range(in_channels):
          channel_weight_ic = weight.detach()[:, ic]
          channel_weight_jc = weight.detach()[:, jc]
          correlation_matrix[ic, jc] = signal_to_noise_ratio(channel_weight_ic,
                                                             channel_weight_jc)
    elif criteria == 'cos':
            for ic in range(in_channels-1):
              for jc in range(in_channels):
                channel_weight_ic = weight.detach()[:, ic]
                channel_weight_jc = weight.detach()[:, jc]
                correlation_matrix[ic, jc] = cosineSimilarity(channel_weight_ic,
                                                              channel_weight_jc)
        # default: Euclide distance
    else:
      for ic in range(in_channels-1):
        for jc in range(ic+1, in_channels):
          channel_weight_ic = weight.detach()[:, ic]
          channel_weight_jc = weight.detach()[:, jc]
          correlation_matrix[ic, jc] = EuclideDistance(channel_weight_ic,
                                                       channel_weight_jc)
          correlation_matrix[jc, ic] = correlation_matrix[ic, jc]

    return correlation_matrix

def get_input_channel_saliency_by_norm(weight, criteria):
    in_channels = weight.shape[1]
    saliencies = []

    # compute the importance for each input channel
    if (criteria == 'L0_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(torch.flatten(channel_weight), 0)
            saliencies.append(saliency.view(1))

    elif (criteria == 'L1_norm'):
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            saliency = torch.linalg.vector_norm(torch.flatten(channel_weight), 1)
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
            low=0, high=weight.shape[1],size=weight.shape[1])
                                      .astype('uint8')).to(0)
        return saliencies
    elif (criteria == 'L2_norm'):
        saliencies = get_input_channel_saliency_by_norm(weight, criteria)
    else:
        correlation_matrix = get_input_channel_similarity(weight, criteria)
        for ic in range(weight.shape[1]):
            saliency = - correlation_matrix[ic, :].sum()
            saliencies.append(saliency.view(1))

    return torch.cat(saliencies)

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
