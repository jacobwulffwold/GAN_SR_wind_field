"""
trainingtricks.py
Based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements some useful methods with the goal of improving GAN training.

"""


import random

import torch

import math


def noisy_labels(
    label_type: bool,
    batch_size: torch.Tensor,
    noise_stddev: torch.Tensor=torch.tensor(0.05),
    false_label_val: torch.Tensor=torch.tensor(0.0),
    true_label_val: torch.Tensor=torch.tensor(1.0),
    val_lower_lim: torch.Tensor=torch.tensor(0.0),
    val_upper_lim: torch.Tensor=torch.tensor(1.0),
    device:torch.device=torch.device("cpu")
) -> torch.Tensor:
    """
    noisy_labels adds gaussian noise to True/False GAN label values,
    but keeps the resulting value within a specified range,
    and returns a tensor of sz batch_size filled with that value.
    @arg label_type: True if representing images perceived as real (not generated), else False
    @arg noise_stddev: gaussian noise stddev
    @arg [false|true]_label_val: label values without noise.
    @arg val_[lower|upper]_lim: thresholds for label val cutoff
    """
    label_val=torch.normal(mean=0.0, std=torch.full(torch.Size([batch_size]), noise_stddev)).to(device)
    if label_type == True:
        label_val += true_label_val
    else:
        label_val += false_label_val
    label_val[label_val>val_upper_lim] = val_upper_lim
    label_val[label_val<val_lower_lim] = val_lower_lim
    return label_val


def instance_noise(
    sigma_base: float, shape: torch.Size, it: int, niter: int, device=torch.device("cpu")
) -> torch.Tensor:
    noise = torch.rand(shape, device=device)  # N(0,1)
    var_desired = sigma_base * (1 - (it - 1) / niter)

    return noise * math.sqrt(var_desired)
