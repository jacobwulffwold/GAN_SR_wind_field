"""
discriminator.py
based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements VGG-style discriminator for 3D data
"""

import torch.nn as nn

from CNN_models.torch_blocks import create_discriminator_block, create_conv_lrelu_layer
import tools.loggingclass as lc


class Discriminator_3D(nn.Module, lc.GlobalLoggingClass):
    """
    VGG Style discriminator
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
        device="cpu",
        number_of_z_layers=10,
        conv_mode: str = "3D",
        use_mixed_precision: bool = False,
        enable_slicing: bool = False,
        dropout_probability: float = 0.0,
    ):
        super(Discriminator_3D, self).__init__()
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
        # self.scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

        remainder_z_layers = [number_of_z_layers]
        for i in range(5):
            if i == 0 and number_of_z_layers <= 19:
                remainder_z_layers.append(number_of_z_layers)
            elif i in {1, 2, 3}:
                remainder_z_layers.append(remainder_z_layers[i])
            else:
                remainder_z_layers.append(
                    remainder_z_layers[i] // 2 + remainder_z_layers[i] % 2
                )

        # 64x64x10 -> 32x32x10
        features.append(
            create_discriminator_block(
                in_channels,
                base_number_of_features,
                feat_kern_size=feat_kern_size,
                lrelu_negative_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=True,
                halve_z_dim=False if number_of_z_layers <= 19 else True,
                number_of_z_layers=remainder_z_layers[0],
                mode=conv_mode,
            )
        )
        # 32x32x10 -> 16x16x10
        features.append(
            create_discriminator_block(
                base_number_of_features,
                base_number_of_features * 2,
                feat_kern_size=feat_kern_size,
                lrelu_negative_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
                halve_z_dim=False,
                number_of_z_layers=remainder_z_layers[1],
                mode=conv_mode,
            )
        )
        # 16x16x10 -> 8x8x10
        features.append(
            create_discriminator_block(
                base_number_of_features * 2,
                base_number_of_features * 4,
                feat_kern_size=feat_kern_size,
                lrelu_negative_slope=slope,
                normalization_type=normalization_type,
                drop_first_norm=False,
                halve_z_dim=False,
                number_of_z_layers=remainder_z_layers[2],
                mode=conv_mode,
            )
        )
        if not enable_slicing:
            # 16x16x10 -> 8x8x10
            features.append(
                create_discriminator_block(
                    base_number_of_features * 4,
                    base_number_of_features * 8,
                    feat_kern_size=feat_kern_size,
                    lrelu_negative_slope=slope,
                    normalization_type=normalization_type,
                    drop_first_norm=False,
                    halve_z_dim=False,
                    number_of_z_layers=remainder_z_layers[3],
                    mode=conv_mode,
                )
            )
            # 8x8x10 -> 4x4x5
            features.append(
                create_discriminator_block(
                    base_number_of_features * 8,
                    base_number_of_features * 8,
                    feat_kern_size=feat_kern_size,
                    lrelu_negative_slope=slope,
                    normalization_type=normalization_type,
                    drop_first_norm=False,
                    halve_z_dim=True,
                    number_of_z_layers=remainder_z_layers[4],
                    mode=conv_mode,
                )
            )
        else:
            # 8x8x10 -> 4x4x10
            features.append(
                create_discriminator_block(
                    base_number_of_features * 4,
                    base_number_of_features * 8,
                    feat_kern_size=feat_kern_size,
                    lrelu_negative_slope=slope,
                    normalization_type=normalization_type,
                    drop_first_norm=False,
                    halve_z_dim=False,
                    number_of_z_layers=remainder_z_layers[3],
                    mode=conv_mode,
                )
            )
            # 8x8x10 -> 4x4x5
            features.append(
                create_conv_lrelu_layer(
                    base_number_of_features * 8,
                    base_number_of_features * 8,
                    feat_kern_size,
                    normalization_type="batch",
                )
            )
            features.append(
                create_conv_lrelu_layer(
                    base_number_of_features * 8,
                    base_number_of_features * 8,
                    feat_kern_size,
                    stride=(1, 1, 2),
                    normalization_type="batch",
                )
            )

        classifier = []
        classifier.append(
            nn.Linear(base_number_of_features * 8 * 4 * 4 * remainder_z_layers[5], 100)
        )
        classifier.append(nn.LeakyReLU(negative_slope=slope))
        classifier.append(nn.Linear(100, 1))

        self.dropout = (
            nn.Dropout2d(p=dropout_probability)
            if conv_mode == "2D"
            else nn.Dropout3d(p=dropout_probability)
        )

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

        self.status_logs.append(f"Discriminator: finished init")

    def forward(self, x):
        x = self.dropout(self.features(x))
        # flatten
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
