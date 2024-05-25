"""
Taken from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/discriminator_arch.py 
Authors: xinntao
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

##############################################################################################################################
##############################################################################################################################

# @ARCH_REGISTRY.register()
class add_attn(nn.Module):

    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

# @ARCH_REGISTRY.register()
class unetCat(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        norm = spectral_norm
        self.convU = norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(input_2, scale_factor=2, mode='bilinear', align_corners=False)

        output_2 = F.leaky_relu(self.convU(input_2), negative_slope=0.2, inplace=True)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y

@ARCH_REGISTRY.register()
class Attention_UNetDiscriminatorSN(nn.Module):
    """Defines a Attention-based U-Net discriminator with spectral normalization (SN)

    It is used in A-ESRGAN: Training Real-World Blind Super-Resolution with Attention-based U-net Discriminators.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(Attention_UNetDiscriminatorSN, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False))

        # Center
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False))

        self.gating = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))

        # attention Blocks
        self.attn_1 = add_attn(x_channels=num_feat * 4, g_channels=num_feat * 4)
        self.attn_2 = add_attn(x_channels=num_feat * 2, g_channels=num_feat * 4)
        self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        gated = F.leaky_relu(self.gating(x3), negative_slope=0.2, inplace=True)

        # Attention
        attn1 = self.attn_1(x2, gated)
        attn2 = self.attn_2(x1, gated)
        attn3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
    
################################################################################################################################
################################################################################################################################

@ARCH_REGISTRY.register()
class PatchGAN(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, kernel_size=4, stride=2, padding=1):
        super(PatchGAN, self).__init__()
        norm = spectral_norm
        self.conv1 = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = norm(nn.Conv2d(in_channels=num_feat, out_channels=num_feat*2, kernel_size=kernel_size, stride=stride, padding=padding))
        self.norm2 = nn.BatchNorm2d(num_features=num_feat*2)
        self.conv3 = norm(nn.Conv2d(in_channels=num_feat*2, out_channels=num_feat*4, kernel_size=kernel_size, stride=stride, padding=padding))
        self.norm3 = nn.BatchNorm2d(num_features=num_feat*4)
        self.conv4 = norm(nn.Conv2d(in_channels=num_feat*4, out_channels=num_feat*8, kernel_size=kernel_size, stride=1, padding=padding))
        self.norm4 = nn.BatchNorm2d(num_features=num_feat*8)
        self.conv5 = nn.Conv2d(in_channels=num_feat*8, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)

        self.act = nn.LeakyReLU(0.2, True)
    
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.norm2(self.conv2(x1)))
        x3 = self.act(self.norm3(self.conv3(x2)))
        x4 = self.act(self.norm4(self.conv4(x3)))
        out = self.conv5(x4)
        return out