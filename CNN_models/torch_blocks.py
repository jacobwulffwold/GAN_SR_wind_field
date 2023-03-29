from torch import nn
import torch


def create_conv_lrelu_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    negative_slope=0.2,
    normalization_type="",
    layer_type=nn.Conv2d,
):
    layers = [
        layer_type(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    ]

    if normalization_type:
        if normalization_type == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif normalization_type == "instance":
            return nn.InstanceNorm2d(out_channels)
        else:
            raise NotImplementedError(f"Unknown norm type {normalization_type}")

    layers.append(nn.LeakyReLU(negative_slope=negative_slope))

    return nn.Sequential(*layers)


class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule

    def forward(self, x):
        return x + self.module(x)


class RDB_Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        lrelu_neg_slope: float = 0.2,
    ):
        super(RDB_Conv, self).__init__()
        self.conv = create_conv_lrelu_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            negative_slope=lrelu_neg_slope,
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """
    Based on: Residual Dense Network for Image Super-Resolution (Zhang et al., 2018)
    This variation supports different depths
    """

    def __init__(
        self,
        in_channels: int,
        growth_channels: int,
        number_of_conv_layers: int,
        lff_kern_size: int = 1,
        lrelu_neg_slope: float = 0.2,
        residual_scaling=0.2,
    ):
        super(RDB, self).__init__()
        self.residual_scaling = residual_scaling
        for i in range(number_of_conv_layers - 1):
            self.add_module(
                "conv{}".format(i),
                RDB_Conv(
                    in_channels + i * growth_channels,
                    growth_channels,
                    lrelu_neg_slope=lrelu_neg_slope,
                ),
            )

            # TODO: ESDRGAN uses 3x3 kernel here. Is it a bug in their code, or intentional?

        # In https://arxiv.org/pdf/1802.08797.pdf it's specified that LFF should have a 1x1 kern.

        if lff_kern_size <= 0 or (lff_kern_size % 2) == 0:
            raise ValueError(
                "LFF kernel size (lff_kern_size) must be an odd number > 0"
            )

        lff_pad = (lff_kern_size - 1) // 2  # no dim change

        self.LFF = nn.Conv2d(
            in_channels + (number_of_conv_layers-1) * growth_channels,
            in_channels,
            kernel_size=lff_kern_size,
            padding=lff_pad,
        )

    def forward(self, x):
        next_x=x.clone()
        for i in range(len(self._modules) - 1):
            next_x = self.__getattr__("conv{}".format(i))(next_x)
        residual = self.LFF(next_x)
        return residual.mul(self.residual_scaling) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        in_channels: int,
        growth_channels: int,
        num_convs: int,
        lff_kern_size: int = 1,
        lrelu_neg_slope: float = 0.2,
        RDB_residual_scaling: float = 0.2,
        RRDB_residual_scaling: float = 0.2,
        number_of_RDBs: int = 3,
    ):
        super(RRDB, self).__init__()
        self.RRDB_residual_scaling = RRDB_residual_scaling

        self.RDBs = [
            RDB(
                in_channels,
                growth_channels,
                num_convs,
                lrelu_neg_slope=lrelu_neg_slope,
                residual_scaling=RDB_residual_scaling,
                lff_kern_size=lff_kern_size,
            )
            for i in range(number_of_RDBs)
        ]
        self.RDBs = nn.Sequential(*self.RDBs)

    def forward(self, x):
        residual = self.RDBs(x)
        return residual.mul(self.RRDB_residual_scaling) + x


def create_UpConv_block(
    in_channels: int, out_channels: int, scale: int, lrelu_neg_slope: float = 0.2
):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="nearest"),
        create_conv_lrelu_layer(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            negative_slope=lrelu_neg_slope,
        ),
    )


def create_discriminator_block(
    in_channels: int,
    out_channels: int,
    feat_kern_size: int = 3,
    lrelu_neg_slope: float = 0.2,
    normalization_type: str = "batch",
    drop_first_norm: bool = False,
):
    if feat_kern_size == 5:
        feat_conv_padding = 2
    elif feat_kern_size == 3:
        feat_conv_padding = 1
    else:
        raise NotImplementedError("Only supported kern sizes are 3 and 5")
    if not drop_first_norm:
        layers = [
            create_conv_lrelu_layer(
                in_channels,
                out_channels,
                kernel_size=feat_kern_size,
                negative_slope=lrelu_neg_slope,
                padding=feat_conv_padding,
                stride=1,
                normalization_type=normalization_type,
            )
        ]
    else:
        layers = [
            create_conv_lrelu_layer(
                in_channels,
                out_channels,
                kernel_size=feat_kern_size,
                negative_slope=lrelu_neg_slope,
                padding=feat_conv_padding,
                stride=1,
            )
        ]

    layers.append(
        create_conv_lrelu_layer(
            out_channels,
            out_channels,
            kernel_size=4,
            negative_slope=lrelu_neg_slope,
            padding=1,
            stride=2,
            normalization_type=normalization_type,
        )
    )

    return nn.Sequential(*layers)
