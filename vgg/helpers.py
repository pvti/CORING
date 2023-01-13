import torch
from torch import nn
from torchprofile import profile_macs
import time

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module,
                   data_width=32,
                   count_nonzero_only=False
                   ) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    """Measure latency of a regular PyTorch models
    """
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency


def get_model_performance(model):
    """Caculate model's performance
    """
    num_params = round(get_num_parameters(model) / 1e6, 2)  # in million
    size = round(get_model_size(model) / MiB, 2)
    # measure on cpu to simulate inference on an edge device
    ori_device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')
    model = model.to('cpu')
    latency = round(measure_latency(model, dummy_input) * 1000, 1)  # in ms
    macs = round(get_model_macs(model, dummy_input) / 1e6)  # in million
    model = model.to(ori_device)

    return size, latency, macs, num_params
