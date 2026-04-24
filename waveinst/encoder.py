import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import caffe2_xavier_init, kaiming_init

from mmdet.registry import MODELS


class PyramidPoolingModule(nn.Module):

    def __init__(self,
                 in_channels,
                 channels=512,
                 sizes=(1, 2, 3, 6),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels + len(sizes) * channels,
                                    in_channels, 1)
        self.act = MODELS.build(act_cfg)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=self.act(stage(feats)),
                size=(h, w),
                mode='bilinear',
                align_corners=False) for stage in self.stages
        ] + [feats]
        out = self.act(self.bottleneck(torch.cat(priors, 1)))
        return out


# SE Block
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AGFM(nn.Module):
    def __init__(self,
                 in_channels1,
                 in_channels2,
                 out_channels):
        super().__init__()
        self.se_fpn = SE_Block(in_channels1)
        self.se_dwt = SE_Block(in_channels2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1, bias = False),
            nn.ReLU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat_fpn, feat_dwt):
        feat_fpn = self.se_fpn(feat_fpn)
        feat_dwt = self.se_dwt(feat_dwt)
        feat_concat = torch.cat([feat_fpn, feat_dwt], dim=1)
        feat_fusion = self.fusion(feat_concat)
        G = self.gate_conv(feat_fusion)
        return G * feat_fpn + (1 - G) * feat_dwt


@MODELS.register_module()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 with_ppm=True,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.with_ppm = with_ppm
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = nn.Conv2d(in_channel, self.num_channels, 1)
            output_conv = nn.Conv2d(
                self.num_channels, self.num_channels, 3, padding=1)
            caffe2_xavier_init(lateral_conv)
            caffe2_xavier_init(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        if self.with_ppm:
            self.ppm = PyramidPoolingModule(
                self.num_channels, self.num_channels // 4, act_cfg=act_cfg)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        kaiming_init(self.fusion)

    def forward(self, features):
        features = features[::-1]
        prev_features = self.fpn_laterals[0](features[0])
        if self.with_ppm:
            prev_features = self.ppm(prev_features)
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:],
                                                  self.fpn_laterals[1:],
                                                  self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [
            F.interpolate(x, size, mode='bilinear', align_corners=False)
            for x in outputs[1:]
        ]
        features = self.fusion(torch.cat(features, dim=1))
        return features


@MODELS.register_module()
class WaveFusionEncoder(nn.Module):
    """
    Wave Fusion Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    4. wavelet-guided enhance
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 with_ppm=True,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.with_ppm = with_ppm
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = nn.Conv2d(in_channel, self.num_channels, 1)
            output_conv = nn.Conv2d(
                self.num_channels, self.num_channels, 3, padding=1)
            caffe2_xavier_init(lateral_conv)
            caffe2_xavier_init(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        if self.with_ppm:
            self.ppm = PyramidPoolingModule(
                self.num_channels, self.num_channels // 4, act_cfg=act_cfg)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        # AGFM
        self.agfm = AGFM(self.num_channels, self.num_channels, self.num_channels)
        kaiming_init(self.fusion)

    def forward(self, features, f_dwt):
        features = features[::-1]
        prev_features = self.fpn_laterals[0](features[0])
        if self.with_ppm:
            prev_features = self.ppm(prev_features)
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:],
                                                  self.fpn_laterals[1:],
                                                  self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [
            F.interpolate(x, size, mode='bilinear', align_corners=False)
            for x in outputs[1:]
        ]
        features = self.fusion(torch.cat(features, dim=1))
        # AGFM
        output_features = self.agfm(features, f_dwt)
        return output_features
