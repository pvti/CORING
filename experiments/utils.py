import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker


def svd(x, rank=1):
    x_unfold = tl.unfold(x, 0)
    u, _, vh = np.linalg.svd(x_unfold, full_matrices=False)
    v = np.transpose(vh)
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
    vbd = np.var(a - b) / (np.var(a) + np.var(b))

    return vbd
