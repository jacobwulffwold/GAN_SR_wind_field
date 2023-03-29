"""
generator.py
based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements the ESRDnet nn.Module
"""

import logging
import math

import torch
import torch.nn as nn

import functools
from CNN_models.torch_blocks import RRDB, create_UpConv_block, SkipConnectionBlock, create_conv_lrelu_layer
import tools.loggingclass as lc


class Generator_2D(nn.Module, lc.GlobalLoggingClass):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        number_of_features: int,
        number_of_RRDBs: int,
        upscale: int = 4,
        hr_kern_size: int = 3,
        number_of_RDB_convs: int = 5,
        RDB_gc: int = 32,
        lff_kern_size: int = 3,
        RDB_residual_scaling: float = 0.2,
        RRDB_residual_scaling: float = 0.2,
        act_type: str = "leakyrelu",
        # device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        device = "cpu"
    ):
        super(Generator_2D, self).__init__()

        slope = 0
        if act_type == "leakyrelu":
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            self.status_logs.append(
                f"Generator: warning: activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)"
            )
            slope = 0.2

        # Low level feature extraction
        feture_conv = nn.Conv2d(
            in_channels, number_of_features, kernel_size=3, padding=1
        )

        RRDBs = [
            RRDB(
                number_of_features,
                RDB_gc,
                number_of_RDB_convs,
                lff_kern_size,
                lrelu_neg_slope=slope,
                RDB_residual_scaling=RDB_residual_scaling,
                RRDB_residual_scaling=RRDB_residual_scaling,
            )
            for i in range(number_of_RRDBs)
        ]

        lr_conv = nn.Conv2d(
            number_of_features, number_of_features, kernel_size=3, padding=1
        )
        # Shortcut from feature_conv to the upsampler
        RRDB_conv_shortcut = SkipConnectionBlock(nn.Sequential(*RRDBs, lr_conv))

        # Upsampling: Upsample+conv combo
        number_of_upsample_layers = math.floor(math.log2(upscale))
        if 2**number_of_upsample_layers != upscale:
            self.status_logs.append(
                f"ESRDnet: warning: upsampling only supported for factors 2^n. Defaulting {upscale} to {2**number_of_upsample_layers}"
            )

        upsampler = [
            create_UpConv_block(
                number_of_features, number_of_features, scale=2, lrelu_neg_slope=slope
            )
            for upsample in range(number_of_upsample_layers)
        ]

        hr_pad = 1
        if hr_kern_size == 5:
            hr_pad = 2
        elif hr_kern_size == 3:
            pass
        else:
            raise NotImplementedError("Only kern sizes 3 and 5 are supported")

        hr_convs = [create_conv_lrelu_layer(number_of_features, number_of_features, kernel_size=hr_kern_size, padding=hr_pad, negative_slope=slope),
            nn.Conv2d(
                number_of_features,
                out_channels,
                kernel_size=hr_kern_size,
                padding=hr_pad,
            ),
        ]

        self.model = nn.Sequential(
            feture_conv, RRDB_conv_shortcut, *upsampler, *hr_convs
        )
        self.status_logs.append(f"Generator: finished init")

    def forward(self, x):
        return self.model(x)
