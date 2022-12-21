import torch
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import SignalNoiseRatio
import tensorly as tl
from tensorly import unfold

tl.set_backend('pytorch')
cos = torch.nn.CosineSimilarity(dim=0)


def cosine(a, b):
    """Caculate cosine similarity
    """

    return abs(cos(torch.flatten(a), torch.flatten(b)))


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

    return torch.var(signal-noise) / torch.var(signal)


def get_u_svd(x):
    """Perform SVD, return 1st column of u
    """
    u, _, _ = torch.linalg.svd(x, full_matrices=False)

    return u[:, 0]


def svd_cos(x, y):
    """Caculate absolute value of cosine of 2 matrix
        x, y: 2d matrix (unfoled from 3d tensor)
    """

    return abs(cos(get_u_svd(x), get_u_svd(y)))


def svd(x, y):
    """Caculate cosine similarity through SVD
        x, y: 3d filter
        First unfold x, y then return cosine
    """
    ufx = unfold(x, 0)
    ufy = unfold(y, 0)

    return svd_cos(ufx, ufy)


def hosvd(x, y):
    """Caculate cosine similarity through HOSVD
        x, y: 3d filter
        First unfold x, y in each dim
        Then return averaged cosine
    """
    sum = 0
    ufx = [0, 0, 0]
    ufy = [0, 0, 0]
    for i in range(3):
        ufx[i] = unfold(x, i)
        ufy[i] = unfold(y, i)
        sum += svd_cos(ufx[i], ufy[i])

    return sum/3
