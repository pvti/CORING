import torch
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import tucker


tl.set_backend("pytorch")


def get_num_unfold(decomposer):
    """Get number of unfolding based on decomposer"""

    return 1 if decomposer == "svd" else 3


def get_u_svd(x):
    """Perform SVD, return 1st column of u"""
    u, _, _ = torch.linalg.svd(x, full_matrices=False)

    return u[:, 0]


def decompose(x, decomposer="hosvd"):
    """Get left_singulars based on decomposer
    svd: u
    hosvd: [u1, u2, u3]
    tucker: [u1, u2, u3]
    """
    if decomposer == "tucker":
        _, factors = tucker(x, rank=[1, 1, 1])

        return factors

    num_unfold = get_num_unfold(decomposer)
    left_singulars = []
    for i in range(num_unfold):
        unfold_i = unfold(x, i)
        left_singular_i = get_u_svd(unfold_i)
        left_singulars.append(left_singular_i)

    return left_singulars
