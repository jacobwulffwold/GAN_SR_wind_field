"""
generator.py
based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements the ESRDnet nn.Module
"""

import math
import torch.nn as nn
import torch
from CNN_models.torch_blocks import (
    RRDB,
    create_UpConv_block,
    SkipConnectionBlock,
    create_conv_lrelu_layer,
    Horizontal_Conv_3D,
    create_UpConv_block,
)
import tools.loggingclass as lc


class Generator_3D(nn.Module, lc.GlobalLoggingClass):
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
        lff_kern_size: int = 1,
        RDB_residual_scaling: float = 0.2,
        RRDB_residual_scaling: float = 0.2,
        act_type: str = "leakyrelu",
        number_of_z_layers: int = 10,
        conv_mode: str = "3D",
        use_mixed_precision: bool = False,
        # device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        device="cpu",
        terrain_number_of_features: int = 16,
        dropout_probability: float = 0.0,
        max_norm:float = 1.0
    ):
        super(Generator_3D, self).__init__()

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
        
        self.max_norm = max_norm

        layer_type = nn.Conv2d if conv_mode == "2D" else nn.Conv3d

        hr_pad = (hr_kern_size - 1) // 2
        # self.scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

        if dropout_probability==None:
            dropout_probability = 0.0

        dropout = (
            nn.Dropout2d(p=dropout_probability)
            if conv_mode == "2D"
            else nn.Dropout3d(p=dropout_probability)
        )

        # Low level feature extraction
        if conv_mode in {"3D", "2D"}:
            feature_conv = create_conv_lrelu_layer(
                in_channels,
                number_of_features,
                3,
                padding=1,
                layer_type=layer_type,
                lrelu=False,
            )
            lr_conv = create_conv_lrelu_layer(
                number_of_features,
                number_of_features,
                3,
                padding=1,
                layer_type=layer_type,
                lrelu_negative_slope=slope,
                lrelu=False,
            )
            hr_convs_w_dropout = [
                create_conv_lrelu_layer(
                    number_of_features + terrain_number_of_features,
                    number_of_features + terrain_number_of_features,
                    kernel_size=hr_kern_size,
                    padding=hr_pad,
                    lrelu_negative_slope=slope,
                    layer_type=layer_type,
                ),
                dropout,
                layer_type(
                    number_of_features + terrain_number_of_features,
                    out_channels,
                    kernel_size=hr_kern_size,
                    padding=hr_pad,
                ),
            ]
            # terrain_conv = create_conv_lrelu_layer(
            #     1,
            #     terrain_number_of_features,
            #     3,
            #     padding=1,
            #     layer_type=layer_type,
            #     lrelu=False,
            # )
            terrain_convs = [create_conv_lrelu_layer(
                1,
                terrain_number_of_features,
                3,
                padding=1,
                layer_type=layer_type,
                lrelu=True,
            ),
            create_conv_lrelu_layer(
                terrain_number_of_features,
                terrain_number_of_features,
                3,
                padding=1,
                layer_type=layer_type,
                lrelu=False,
            ),
            ]


        elif conv_mode == "horizontal_3D":
            feature_conv = Horizontal_Conv_3D(
                in_channels,
                number_of_features,
                3,
                number_of_z_layers=number_of_z_layers,
                lrelu=False,
            )
            lr_conv = Horizontal_Conv_3D(
                number_of_features,
                number_of_features,
                3,
                lrelu_negative_slope=slope,
                number_of_z_layers=number_of_z_layers,
                lrelu=False,
            )
            hr_convs_w_dropout = [
                Horizontal_Conv_3D(
                    number_of_features + terrain_number_of_features,
                    number_of_features + terrain_number_of_features,
                    kernel_size=hr_kern_size,
                    lrelu_negative_slope=slope,
                    number_of_z_layers=number_of_z_layers,
                ),
                dropout,
                Horizontal_Conv_3D(
                    number_of_features + terrain_number_of_features,
                    out_channels,
                    kernel_size=hr_kern_size,
                    number_of_z_layers=number_of_z_layers,
                    lrelu=False,
                ),
            ]
            terrain_conv = Horizontal_Conv_3D(
                1,
                terrain_number_of_features,
                3,
                number_of_z_layers=number_of_z_layers,
                lrelu=False,
            )

        else:
            raise ValueError(f"Conv mode {conv_mode} not implemented")

        RRDBs = [
            RRDB(
                number_of_features,
                RDB_gc,
                number_of_RDB_convs,
                lff_kern_size,
                lrelu_negative_slope=slope,
                RDB_residual_scaling=RDB_residual_scaling,
                RRDB_residual_scaling=RRDB_residual_scaling,
                mode=conv_mode,
            )
            for i in range(number_of_RRDBs)
        ]

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
                number_of_features,
                number_of_features,
                scale=2,
                lrelu_negative_slope=slope,
                number_of_z_layers=number_of_z_layers,
                mode=conv_mode,
            )
            for upsample in range(number_of_upsample_layers)
        ]

        self.model = nn.Sequential(feature_conv, RRDB_conv_shortcut, *upsampler)
        self.hr_convs = nn.Sequential(*hr_convs_w_dropout)
        self.terrain_convs = nn.Sequential(*terrain_convs)
        self.status_logs.append(f"Generator: finished init")

    def forward(self, x, Z):
        x = self.model(x)
        Z = self.terrain_convs(Z)
        x = torch.cat((x, Z), dim=1)
        return self.hr_convs(x)
