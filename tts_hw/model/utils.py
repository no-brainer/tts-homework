import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_padding(kernel_size: int, dilation: int = 1) -> int:
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


def get_mask_from_padding(x, padding_idx=0):
    return (x != padding_idx).float()


def get_mask_from_lengths(lengths, max_len):
    rng = torch.arange(max_len)
    return rng.unsqueeze(0) < lengths.unsqueeze(1)
