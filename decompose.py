import torch
import tensorly as tl
from tensorly import unfold

tl.set_backend('pytorch')


def get_u_svd(x):
    """Perform SVD, return 1st column of u
    """
    u, _, _ = torch.linalg.svd(x, full_matrices=False)

    return u[:, 0]


def decomposition(x, decomposer='hosvd'):
    if (decomposer == 'hosvd'):
        left_singulars = []
        for i in range(3):
            unfold_i = unfold(x, i)
            left_singular_i = get_u_svd(unfold_i)
            left_singulars.append(left_singular_i)

        return left_singulars
