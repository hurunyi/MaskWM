import torch
from torch import nn
import torch.nn.functional as F
from .Noise import Noise


def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return ((images - 0.5) * 2).clamp(-1, 1)


def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        # replace BN with GN
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.bn_s1 = LayerNorm(out_ch)
        self.bn_s1 = nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=1e-6, affine=True)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


### U^2-Net small ###
class U2NETP(nn.Module):
    """ https://github.com/xuebinqin/U-2-Net """
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0


class JND(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """

    def __init__(
        self,
        preprocess=lambda x: x,
        postprocess=lambda x: x,
        in_channels=1,
        out_channels=3,
    ) -> None:
        super(JND, self).__init__()

        # setup input and output methods
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups = self.in_channels

        # create kernels
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(
            [[1., 2., 1.],
             [0., 0., 0.],
             [-1., -2., -1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.tensor(
            [[1., 1., 1., 1., 1.],
             [1., 2., 2., 2., 1.],
             [1., 2., 0., 2., 1.],
             [1., 2., 2., 2., 1.],
             [1., 1., 1., 1., 1.]]
        ).unsqueeze(0).unsqueeze(0)

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)
        kernel_lum = kernel_lum.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_lum = nn.Conv2d(3, 3, kernel_size=(5, 5), padding=2, bias=False, groups=groups)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum, requires_grad=False)

        # setup pre and post processing
        self.preprocess = preprocess
        self.postprocess = postprocess

    def jnd_la(self, x, alpha=1.0, eps=1e-5):
        """ Luminance masking: x must be in [0,255] """
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum] / 127 + eps))
        la[~mask_lum] = 3 / 128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117, eps=1e-5):
        """ Contrast masking: x must be in [0,255] """
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        cm = 16 * cm ** 2.4 / (cm ** 2 + 26 ** 2)
        return beta * cm

    # @torch.no_grad()
    def heatmaps(
        self,
        imgs: torch.Tensor,
        clc: float = 0.3
    ) -> torch.Tensor:
        """ imgs must be in [0,1] after preprocess """
        imgs = 255 * imgs
        rgbs = torch.tensor([0.299, 0.587, 0.114])
        if self.in_channels == 1:
            imgs = rgbs[0] * imgs[..., 0:1, :, :] + rgbs[1] * imgs[..., 1:2, :, :] + rgbs[2] * imgs[..., 2:3, :, :]  # luminance: b 1 h w
        la = self.jnd_la(imgs)
        cm = self.jnd_cm(imgs)
        hmaps = torch.clamp_min(la + cm - clc * torch.minimum(la, cm), 0)  # b 1 or 3 h w
        if self.out_channels == 3 and self.in_channels == 1:
            hmaps = hmaps.repeat(1, 3, 1, 1)  # b 3 h w
        elif self.out_channels == 1 and self.in_channels == 3:
            hmaps = torch.sum(hmaps / 3, dim=1, keepdim=True)  # b 1 h w
        return hmaps / 255

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0, blue: bool = False) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        imgs_w = self.preprocess(imgs_w)
        hmaps = self.heatmaps(imgs, clc=0.3)
        if blue:
            hmaps[:, 0] = hmaps[:, 0] * 0.75
            hmaps[:, 1] = hmaps[:, 1] * 0.50
            hmaps[:, 2] = hmaps[:, 2] * 1.00
        imgs_w = imgs + alpha * hmaps * (imgs_w - imgs)
        return self.postprocess(imgs_w)


def GroupNorm32(channels):
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
    raise ValueError("Invalid number of channels for GroupNorm!")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type="batch"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            normalization(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type="batch"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False, norm_type="batch"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, norm_type=norm_type))
        self.down1 = (Down(64, 128, norm_type=norm_type))
        self.down2 = (Down(128, 256, norm_type=norm_type))
        self.down3 = (Down(256, 512, norm_type=norm_type))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, norm_type=norm_type))
        self.up1 = (Up(1024, 512 // factor, bilinear, norm_type=norm_type))
        self.up2 = (Up(512, 256 // factor, bilinear, norm_type=norm_type))
        self.up3 = (Up(256, 128 // factor, bilinear, norm_type=norm_type))
        self.up4 = (Up(128, 64, bilinear, norm_type=norm_type))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.function = F.interpolate
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.function(x, size=self.size, scale_factor=self.scale_factor)


class ConvNormRelu(nn.Module):
    """
    A sequence of Convolution, Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, normalization, stride):
        super(ConvNormRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            normalization(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        message_length=16,
        in_channels=4,
        mask_channel=1,
        tp_channels=3,
        channels=64,
        norm_type="batch",
        **kwargs
    ):
        super(Encoder, self).__init__()

        self.mask_channel = mask_channel
        self.message_length = message_length
        
        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.message_linear = nn.Linear(message_length, message_length * message_length)
        self.message_to_tp_layer = nn.Sequential(
            ConvNormRelu(1, tp_channels, normalization, stride=1),
            ConvNormRelu(tp_channels, tp_channels, normalization, stride=1),
            ConvNormRelu(tp_channels, tp_channels, normalization, stride=1),
        )
        self.unet = UNet(
            n_channels=in_channels + tp_channels + mask_channel,
            n_classes=channels,
            norm_type=norm_type
        )
        self.final_layer = nn.Conv2d(channels + in_channels, in_channels, kernel_size=1)
        self.jnd = JND(preprocess=denormalize, postprocess=normalize)
        
    def forward(self, image, message, mask=None, use_jnd=True, tp=None, jnd_factor=1.0, blue=False):
        bsz, _, H, W = image.shape

        if tp is None:
            # Message Processor
            message_expanded = self.message_linear(message)
            message_expanded = message_expanded.view(-1, 1, self.message_length, self.message_length)
            message_image = F.interpolate(message_expanded, size=(H, W), mode='nearest')
            tp = self.message_to_tp_layer(message_image)

        # concatenate
        if self.mask_channel:
            if mask is None:
                mask = torch.ones((bsz, 1, H, W)).to(image.device)
            else:
                if mask.dim() != 4:
        	    mask = mask.unsqueeze(0)
                mask = mask.repeat(bsz, 1, 1, 1)
            concat = torch.cat([image, tp, mask], dim=1)
        else:
            concat = torch.cat([image, tp], dim=1)
        
        wm_image = self.unet(concat)
        concat_final = torch.cat([wm_image, image], dim=1)
        wm_image = self.final_layer(concat_final)
        if use_jnd:
            wm_image = self.jnd(image, wm_image, jnd_factor, blue)

        return wm_image


class Decoder(nn.Module):
    def __init__(
        self,
        message_length=16,
        in_channels=3,
        tp_channels=3,
        mask_channel=1,
        channels=64,
        norm_type="batch",
        **kwargs
    ):
        super(Decoder, self).__init__()
        self.tp_channels = tp_channels
        self.mask_channel = mask_channel
        self.message_length = message_length

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.image_to_mask_layer = U2NETP(3, mask_channel)
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            normalization(channels),
            nn.ReLU(inplace=True),
        )
        self.keep_layers = UNet(n_channels=channels, n_classes=channels, norm_type=norm_type)        
        self.tp_pre_layer = nn.Sequential(
            nn.Conv2d(channels, tp_channels, 3, 1, 1, bias=False),
            normalization(tp_channels),
            nn.ReLU(inplace=True),
        )
        self.tp_to_message_layer = nn.Sequential(
            ConvNormRelu(tp_channels, tp_channels, normalization, stride=1),
            ConvNormRelu(tp_channels, tp_channels, normalization, stride=1),
            ConvNormRelu(tp_channels, 1, normalization, stride=1),
        )
        self.message_linear = nn.Linear(message_length * message_length, message_length)

    def forward(self, image, mask=None, tp=None):
        
        mask_pred = self.image_to_mask_layer(image)
        if mask is not None:
            image = image * mask
        else:
            image = image * (mask_pred.gt(0.5).float().mean(dim=1, keepdim=True) > 0).float()
        x = self.first_layers(image)
        y = self.keep_layers(x)
        tp_pred = self.tp_pre_layer(y)
        
        if tp is not None:
            tp_pred = tp

        tp_pred = self.tp_to_message_layer(tp_pred)
        message = F.interpolate(tp_pred, size=(self.message_length, self.message_length), mode='nearest')
        message = message.view(message.shape[0], -1)
        message = self.message_linear(message)

        return message, mask_pred


class WatermarkModel(nn.Module):
    def __init__(
        self,
        wm_enc_config,
        wm_dec_config,
        noise_layers="Identity()",
    ):
        super(WatermarkModel, self).__init__()

        self.encoder = Encoder(**wm_enc_config)
        self.decoder = Decoder(**wm_dec_config)
        self.noise = Noise(noise_layers)
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config

    def forward(
        self,
        image,
        message,
        mask=None,
        use_jnd=True,
        vae=None,
    ):
        encoded_image = self.encoder(image, message, mask, use_jnd)
        
        if mask is not None:
            noised_image = encoded_image * mask + image * (1 - mask)
        else:
            noised_image = encoded_image
        
        if vae is not None:
            noised_image, mask_gt = vae(noised_image, mask)
        else:
            noised_image, mask_gt = self.noise(noised_image, mask)
        
        decoded_message, mask_pd = self.decoder(noised_image, mask)
        
        return encoded_image, noised_image, decoded_message, mask_gt, mask_pd
