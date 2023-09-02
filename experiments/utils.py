import torch
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import unfold


tl.set_backend("pytorch")


def svd(x, rank=1):
    x_unfold = unfold(x, 0)
    u, _, vh = torch.linalg.svd(x_unfold, full_matrices=False)
    v = torch.transpose(vh, 0, 1)
    factors = [u[:, :rank], v[:, :rank]]

    return factors


def reshape_decompose(x, rank=1):
    x_rs = x.reshape(x.size(0), -1)
    _, factors = tucker(x_rs, rank=[rank, rank])

    return factors


def tensor_decompose(x, rank=1):
    _, factors = tucker(x, rank=[rank, rank, rank])

    return factors


def VBD(a, b):
    """Caculate variance based distance"""
    vbd = torch.var(a - b) / (torch.var(a) + torch.var(b))

    return vbd
