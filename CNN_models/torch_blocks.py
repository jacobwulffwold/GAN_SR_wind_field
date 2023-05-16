from torch import nn
import torch


def create_conv_lrelu_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=1,
    lrelu_negative_slope=0.2,
    normalization_type="",
    layer_type=nn.Conv2d,
    lrelu=True,
):
    layers = [
        layer_type(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    ]

    if normalization_type:
        if normalization_type == "batch":
            if layer_type == nn.Conv2d:
                layers.append(nn.BatchNorm2d(out_channels))
            elif layer_type == nn.Conv3d:
                layers.append(nn.BatchNorm3d(out_channels))
        elif normalization_type == "instance":
            if layer_type == nn.Conv2d:
                layers.append(nn.InstanceNorm2d(out_channels))
            elif layer_type == nn.Conv3d:
                layers.append(nn.InstanceNorm3d(out_channels))
        else:
            raise NotImplementedError(f"Unknown norm type {normalization_type}")

    if lrelu:
        layers.append(nn.LeakyReLU(negative_slope=lrelu_negative_slope))

    return nn.Sequential(*layers)


class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule

    def forward(self, x):
        return x + self.module(x)


def forward_horizontal_convs(
    x,
    convs: list,
    vertical_kernel_size: int = 3,
    vertical_stride: int = 1,
    vertical_padding: int = 1,
):
    x = torch.nn.functional.pad(
        x, (vertical_padding, vertical_padding, 0, 0, 0, 0, 0, 0, 0, 0), "constant", 0
    )
    for i in range(
        vertical_kernel_size // 2,
        vertical_stride * (len(convs) + (vertical_kernel_size - 1) // 2),
        vertical_stride,
    ):
        this_it = (i - vertical_kernel_size // 2) // vertical_stride

        this_slice = x[
            :,
            :,
            :,
            :,
            i - vertical_kernel_size // 2 : i + (vertical_kernel_size - 1) // 2 + 1,
        ]

        if i == vertical_kernel_size // 2:
            output = convs[this_it](this_slice)
            y = torch.zeros(
                (
                    output.shape[0],
                    output.shape[1],
                    output.shape[2],
                    output.shape[3],
                    len(convs),
                ),
                device=x.device,
            )
            y[:, :, :, :, this_it] = output.squeeze()
        else:
            y[:, :, :, :, this_it] = convs[this_it](this_slice).squeeze()

    return y


class Horizontal_Conv_3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        lrelu_negative_slope: float = 0.2,
        number_of_z_layers=10,
        lrelu=True,
        normalization_type: str = "",  # "batch" or "instance"
        stride: int = 1,
        padding: int = (-1, -1, -1),
    ):
        if padding == (-1, -1, -1):
            padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2, 0)
            self.vertical_padding = (kernel_size - 1) // 2
        else:
            self.vertical_padding = padding if type(padding) == int else padding[2]
            padding = (
                (padding, padding, 0)
                if type(padding) == int
                else (padding[0], padding[1], 0)
            )

        self.vertical_stride = stride if type(stride) == int else stride[2]
        self.vertical_kernel_size = (
            kernel_size if type(kernel_size) == int else kernel_size[2]
        )

        super(Horizontal_Conv_3D, self).__init__()
        self.convs = nn.ModuleList(
            [
                create_conv_lrelu_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    lrelu_negative_slope=lrelu_negative_slope,
                    stride=stride,
                    layer_type=nn.Conv3d,
                    lrelu=lrelu,
                    normalization_type=normalization_type,
                )
                for i in range(
                    (
                        number_of_z_layers
                        - self.vertical_kernel_size
                        + 2 * self.vertical_padding
                    )
                    // self.vertical_stride
                    + 1
                )
            ]
        )

    def forward(self, x):
        return forward_horizontal_convs(
            x,
            self.convs,
            vertical_kernel_size=self.vertical_kernel_size,
            vertical_padding=self.vertical_padding,
            vertical_stride=self.vertical_stride,
        )


class RDB_Horizontal_Conv_3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        lrelu_negative_slope: float = 0.2,
        number_of_z_layers=10,
    ):
        super(RDB_Horizontal_Conv_3D, self).__init__()
        self.convs = nn.ModuleList(
            [
                create_conv_lrelu_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2, 0),
                    lrelu_negative_slope=lrelu_negative_slope,
                    layer_type=nn.Conv3d,
                )
                for i in range(number_of_z_layers)
            ]
        )
        self.out_channels = out_channels
        self.vertical_padding = (kernel_size - 1) // 2

    def forward(self, x):
        out = forward_horizontal_convs(
            x, self.convs, vertical_padding=self.vertical_padding
        )
        return torch.cat((x, out), 1)


class RDB_Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        lrelu_negative_slope: float = 0.2,
        layer_type=nn.Conv2d,
    ):
        super(RDB_Conv, self).__init__()
        self.conv = create_conv_lrelu_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            lrelu_negative_slope=lrelu_negative_slope,
            layer_type=layer_type,
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
        lrelu_negative_slope: float = 0.2,
        residual_scaling=0.2,
        mode="2D",
    ):
        super(RDB, self).__init__()
        self.residual_scaling = residual_scaling
        for i in range(number_of_conv_layers - 1):
            if mode == "2D":
                self.add_module(
                    "conv{}".format(i),
                    RDB_Conv(
                        in_channels + i * growth_channels,
                        growth_channels,
                        lrelu_negative_slope=lrelu_negative_slope,
                    ),
                )
            elif mode == "horizontal_3D":
                self.add_module(
                    "conv{}".format(i),
                    torch.jit.script(
                        RDB_Horizontal_Conv_3D(
                            in_channels + i * growth_channels,
                            growth_channels,
                            lrelu_negative_slope=lrelu_negative_slope,
                        )
                    ),
                )
            elif mode == "3D":
                self.add_module(
                    "conv{}".format(i),
                    torch.jit.script(
                        RDB_Conv(
                            in_channels + i * growth_channels,
                            growth_channels,
                            lrelu_negative_slope=lrelu_negative_slope,
                            layer_type=nn.Conv3d,
                        )
                    ),
                )
            else:
                raise NotImplementedError(f"Unknown RDB mode {mode}")

        if lff_kern_size <= 0 or (lff_kern_size % 2) == 0:
            raise ValueError(
                "LFF kernel size (lff_kern_size) must be an odd number > 0"
            )

        lff_pad = (lff_kern_size - 1) // 2  # no dim change

        self.LFF = nn.Conv3d(
            in_channels + (number_of_conv_layers - 1) * growth_channels,
            in_channels,
            kernel_size=lff_kern_size,
            padding=lff_pad,
        )

    def forward(self, x):
        next_x = x.clone()
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
        lrelu_negative_slope: float = 0.2,
        RDB_residual_scaling: float = 0.2,
        RRDB_residual_scaling: float = 0.2,
        number_of_RDBs: int = 3,
        mode="2D",
    ):
        super(RRDB, self).__init__()
        self.RRDB_residual_scaling = RRDB_residual_scaling

        self.RDBs = [
            RDB(
                in_channels,
                growth_channels,
                num_convs,
                lrelu_negative_slope=lrelu_negative_slope,
                residual_scaling=RDB_residual_scaling,
                lff_kern_size=lff_kern_size,
                mode=mode,
            )
            for i in range(number_of_RDBs)
        ]
        self.RDBs = nn.Sequential(*self.RDBs)

    def forward(self, x):
        residual = self.RDBs(x)
        return residual.mul(self.RRDB_residual_scaling) + x


def create_UpConv_block(
    in_channels: int,
    out_channels: int,
    scale: int,
    lrelu_negative_slope: float = 0.2,
    mode="2D",
    number_of_z_layers=10,
):
    layer_type, scale_factor = (
        (nn.Conv2d, scale) if mode == "2D" else (nn.Conv3d, (scale, scale, 1))
    )

    if mode in {"2D", "3D"}:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            create_conv_lrelu_layer(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                lrelu_negative_slope=lrelu_negative_slope,
                layer_type=layer_type,
            ),
        )
    elif mode == "horizontal_3D":
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            Horizontal_Conv_3D(
                in_channels,
                out_channels,
                kernel_size=3,
                lrelu_negative_slope=lrelu_negative_slope,
                number_of_z_layers=number_of_z_layers,
            ),
        )
    else:
        raise NotImplementedError(f"Unknown UpConv mode {mode}")


def create_discriminator_block(
    in_channels: int,
    out_channels: int,
    feat_kern_size: int = 3,
    lrelu_negative_slope: float = 0.2,
    normalization_type: str = "batch",
    drop_first_norm: bool = False,
    mode: str = "2D",
    number_of_z_layers: int = 10,
    halve_z_dim: bool = True,
):
    if feat_kern_size == 5:
        feat_conv_padding = 2
    elif feat_kern_size == 3:
        feat_conv_padding = 1
    else:
        raise NotImplementedError("Only supported kern sizes are 3 and 5")

    layer_type = nn.Conv2d if mode == "2D" else nn.Conv3d

    if not drop_first_norm:
        if mode in {"2D", "3D"}:
            layers = [
                create_conv_lrelu_layer(
                    in_channels,
                    out_channels,
                    kernel_size=feat_kern_size,
                    lrelu_negative_slope=lrelu_negative_slope,
                    padding=feat_conv_padding,
                    stride=1,
                    normalization_type=normalization_type,
                    layer_type=layer_type,
                )
            ]
        elif mode == "horizontal_3D":
            layers = [
                Horizontal_Conv_3D(
                    in_channels,
                    out_channels,
                    kernel_size=feat_kern_size,
                    lrelu_negative_slope=lrelu_negative_slope,
                    padding=feat_conv_padding,
                    stride=1,
                    normalization_type=normalization_type,
                    number_of_z_layers=number_of_z_layers,
                )
            ]
        else:
            raise NotImplementedError(
                "Only supported modes are 2D, 3D and horizontal_3D"
            )
    else:
        if mode in {"2D", "3D"}:
            layers = [
                create_conv_lrelu_layer(
                    in_channels,
                    out_channels,
                    kernel_size=feat_kern_size,
                    lrelu_negative_slope=lrelu_negative_slope,
                    padding=feat_conv_padding,
                    stride=1,
                    layer_type=layer_type,
                )
            ]
        elif mode == "horizontal_3D":
            layers = [
                Horizontal_Conv_3D(
                    in_channels,
                    out_channels,
                    kernel_size=feat_kern_size,
                    lrelu_negative_slope=lrelu_negative_slope,
                    padding=feat_conv_padding,
                    stride=1,
                    number_of_z_layers=number_of_z_layers,
                )
            ]
        else:
            raise NotImplementedError(
                "Only supported modes are 2D, 3D and horizontal_3D"
            )

    if mode == "2D":
        layers.append(
            create_conv_lrelu_layer(
                out_channels,
                out_channels,
                kernel_size=4,
                lrelu_negative_slope=lrelu_negative_slope,
                padding=1,
                stride=2,
                normalization_type=normalization_type,
            )
        )
    else:
        if halve_z_dim:
            if mode == "3D":
                layers.append(
                    create_conv_lrelu_layer(
                        out_channels,
                        out_channels,
                        kernel_size=(4, 4, feat_kern_size),
                        lrelu_negative_slope=lrelu_negative_slope,
                        padding=1,
                        stride=2,
                        normalization_type=normalization_type,
                        layer_type=layer_type,
                    )
                )
            else:
                layers.append(
                    Horizontal_Conv_3D(
                        out_channels,
                        out_channels,
                        kernel_size=4,
                        lrelu_negative_slope=lrelu_negative_slope,
                        padding=1,
                        stride=2,
                        normalization_type=normalization_type,
                        number_of_z_layers=number_of_z_layers,
                    )
                )
        else:
            if mode == "3D":
                layers.append(
                    create_conv_lrelu_layer(
                        out_channels,
                        out_channels,
                        kernel_size=(4, 4, feat_kern_size),
                        lrelu_negative_slope=lrelu_negative_slope,
                        padding=(1, 1, 1),
                        stride=(2, 2, 1),
                        normalization_type=normalization_type,
                        layer_type=layer_type,
                    )
                )
            else:
                layers.append(
                    Horizontal_Conv_3D(
                        out_channels,
                        out_channels,
                        kernel_size=(4, 4, feat_kern_size),
                        lrelu_negative_slope=lrelu_negative_slope,
                        padding=(1, 1, 1),
                        stride=(2, 2, 1),
                        normalization_type=normalization_type,
                        number_of_z_layers=number_of_z_layers,
                    )
                )

    return nn.Sequential(*layers)
