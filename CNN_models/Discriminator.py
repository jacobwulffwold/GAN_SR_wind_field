"""
discriminator.py
based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements VGG-style discriminators for different input resolutions.
"""

import torch
import torch.nn as nn

from CNN_models.torch_blocks import create_discriminator_block
import tools.loggingclass as lc


class Discriminator_2D(nn.Module, lc.GlobalLoggingClass):
    """
    VGG Style discriminator for 128x128 images
    Based on,
    Recovering Realistic Texture in Image Super-resolution
     by Deep Spatial Feature Transform (Wang et al.)
    """

    def __init__(
        self,
        in_channels: int,
        base_number_of_features: int,
        feat_kern_size: int = 3,
        normalization_type: str = "batch",
        act_type: str = "leakyrelu",
        mode="CNA",
        # device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        device = "cpu"
    ):
        super(Discriminator_2D, self).__init__()
        self.base_number_of_features = base_number_of_features
        slope = 0
        if act_type == "leakyrelu":
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            self.status_logs.append(
                f"Discriminator: warning: activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)"
            )
            slope = 0.2

        features = []

        # 128x128 -> 64x64
        features.append(
            create_discriminator_block(
                in_channels,
                base_number_of_features,
                feat_kern_size=feat_kern_size,
                lrelu_neg_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=True,
            )
        )
        # 64x64 -> 32x32
        features.append(
            create_discriminator_block(
                base_number_of_features,
                base_number_of_features * 2,
                feat_kern_size=feat_kern_size,
                lrelu_neg_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
            )
        )
        # 32x32 -> 16x16
        features.append(
            create_discriminator_block(
                base_number_of_features * 2,
                base_number_of_features * 4,
                feat_kern_size=feat_kern_size,
                lrelu_neg_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
            )
        )
        # 16x16 -> 8x8
        features.append(
            create_discriminator_block(
                base_number_of_features * 4,
                base_number_of_features * 8,
                feat_kern_size=feat_kern_size,
                lrelu_neg_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
            )
        )
        # 8x8 -> 4x4
        features.append(
            create_discriminator_block(
                base_number_of_features * 8,
                base_number_of_features * 8,
                feat_kern_size=feat_kern_size,
                lrelu_neg_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
            )
        )
        # Chans: base_nf*8
        # Dims: 4x4 pixels
        # -> 100 nodes
        classifier = []
        classifier.append(nn.Linear(base_number_of_features * 8 * 4 * 4, 100))
        classifier.append(nn.LeakyReLU(negative_slope=slope))
        classifier.append(nn.Linear(100, 1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

        self.status_logs.append(f"Discriminator: finished init")

    def forward(self, x):
        x = self.features(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
