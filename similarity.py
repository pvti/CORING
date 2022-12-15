import torch
from torch import nn
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import SignalNoiseRatio


def cosine(a, b):
    """Caculate cosine similarity
    """
    cos = torch.nn.CosineSimilarity(dim=0)

    return cos(torch.flatten(a), torch.flatten(b))


def Euclide(a, b):
    """Caculate Euclide distance
    """

    return torch.dist(a, b)


def Manhattan(a, b):
    """Caculate Manhattan similarity
    """

    return torch.dist(a, b, 1)


def Pearson(a, b):
    """Caculate Pearson similarity
    """

    return pearson_corrcoef(torch.flatten(a), torch.flatten(b))

'''
def SNR(signal, noise):
    """Caculate signal to noise ratio
    """
    snr = SignalNoiseRatio().to(0)

    return snr(signal, noise)
'''

def SNR(signal, noise):
    """Caculate signal to noise ratio
    """
    #snr = SignalNoiseRatio().to(0)

    return torch.var(signal-noise) / torch.var(signal)

