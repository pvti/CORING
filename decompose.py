import torch
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import tucker


tl.set_backend("pytorch")


def get_num_unfold(decomposer):
    """Get number of unfolding based on decomposer"""

    return 1 if decomposer == "svd" else 3


def get_u_svd(x, rank=1):
    """Perform SVD, return rank first columns of u"""
    u, _, _ = torch.linalg.svd(x, full_matrices=False)

    return u[:, :rank]


def decompose(x, decomposer="hosvd", rank=1, mode=0):
    """Get left_singulars based on decomposer and rank
    svd: u
    hosvd: [u1, u2, u3]
    tucker: [u1, u2, u3]
    """
    if decomposer == "tucker":
        _, factors = tucker(x, rank=[rank, rank, rank])

        return factors

    elif decomposer == "hosvd":
        num_unfold = get_num_unfold(decomposer)
        left_singulars = []
        for i in range(num_unfold):
            unfold_i = unfold(x, i)
            left_singular_i = get_u_svd(unfold_i, rank=rank)
            left_singulars.append(left_singular_i)

    elif decomposer == "svd":
        unfold_mode = unfold(x, mode)
        left_singular_i = get_u_svd(unfold_mode, rank=rank)
        left_singulars = [left_singular_i]

    return left_singulars
