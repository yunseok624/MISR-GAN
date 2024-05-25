"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
Authors: XPixelGroup
"""
import time
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer
   
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
#         y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class ResidualDenseBlock_CA(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock_CA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_feat, 1)

        self.ca = CoordAtt(num_feat, num_feat)

        self.prelu = nn.PReLU()

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.prelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.prelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.ca(x4)
        return x5 * 0.2 + x

class RRDB_CA(nn.Module):

    def __init__(self, num_feat, num_grow_ch):
        super(RRDB_CA, self).__init__()
        self.rdb1 = ResidualDenseBlock_CA(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_CA(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_CA(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

@ARCH_REGISTRY.register()
class SSR_RRDBNet_CA(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_grow_ch=32, num_block=23):
        super(SSR_RRDBNet_CA, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB_CA, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.prelu = nn.PReLU()
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        if self.scale in [2, 4]:
            feat = self.prelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))

        if self.scale == 3:
            feat = self.prelu(self.conv_up2(F.interpolate(feat, scale_factor=3, mode='nearest')))
        
        if self.scale == 4:
            feat = self.prelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.prelu(self.conv_hr(feat)))
        return self.tanh(out)

##############################################################################################################################

class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_feat, 1, 1, 1)

        self.prelu = nn.PReLU()

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.prelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.prelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x

class RRDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x
    
@ARCH_REGISTRY.register()
class SSR_RRDBNet(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(SSR_RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.prelu = nn.PReLU()
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        if self.scale in [2, 4]:
            feat = self.prelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))

        if self.scale == 3:
            feat = self.prelu(self.conv_up2(F.interpolate(feat, scale_factor=3, mode='nearest')))
        
        if self.scale == 4:
            feat = self.prelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.prelu(self.conv_hr(feat)))
        return self.tanh(out)