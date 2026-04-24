import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import caffe2_xavier_init, kaiming_init
from torch.nn import init

from mmdet.registry import MODELS

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

def _make_stack_3x3_convs(num_convs,
                          in_channels,
                          out_channels,
                          act_cfg=dict(type='ReLU', inplace=True)):
    convs = []
    for _ in range(num_convs):
        convs.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(MODELS.build(act_cfg))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_convs=4,
                 num_masks=100,
                 num_classes=80,
                 kernel_dim=128,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        num_masks = num_masks
        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim,
                                                act_cfg)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob,
                                  features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_convs=4,
                 kernel_dim=128,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim,
                                                act_cfg)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        kaiming_init(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


class DRMaskBranch(nn.Module):
    """
    DRMaskBranch
    D(Dynamic) R(Refine) Mask Branch
    """

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_convs=4,
                 kernel_dim=128,
                 scale=2,
                 style='lp',
                 groups=4,
                 dyscope=False,
                 insert_ind=(1, 3),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim, act_cfg)
        self.dysample1 = DySample(dim, scale=scale, style=style, groups=groups, dyscope=dyscope)
        self.dysample2 = DySample(dim, scale=scale, style=style, groups=groups, dyscope=dyscope)
        self.insert_ind = insert_ind
        if insert_ind:
            if min(insert_ind) < 0:
                raise ValueError("insert_ind must not lower than zero")
            if max(insert_ind) > num_convs - 1:
                raise ValueError(f"insert_ind must not larger than {num_convs - 1}")
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        normal_init(self.dysample1)
        normal_init(self.dysample2)
        kaiming_init(self.projection)

    def forward(self, features):
        # mask features (default: conv0+relu ——> conv1+relu ——> dysample1 ——> conv2+relu ——> conv3+relu ——> dysample2)
        if self.insert_ind:
            for i, layer in enumerate(self.mask_convs):
                features = layer(features)
                if i == self.insert_ind[0] * 2 + 1:
                    features = self.dysample1(features)
                if i == self.insert_ind[1] * 2 + 1:
                    features = self.dysample2(features)
        else:
            features = self.mask_convs(features)
        return self.projection(features)


@MODELS.register_module()
class BaseIAMDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 ins_dim=256,
                 ins_conv=4,
                 mask_dim=256,
                 mask_conv=4,
                 kernel_dim=128,
                 scale_factor=2.0,
                 output_iam=False,
                 num_masks=100,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels  # ENCODER.NUM_CHANNELS + 2

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(
            in_channels,
            dim=ins_dim,
            num_convs=ins_conv,
            num_masks=num_masks,
            num_classes=num_classes,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)
        self.mask_branch = MaskBranch(
            in_channels,
            dim=mask_dim,
            num_convs=mask_conv,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel,
                               mask_features.view(B, C,
                                                  H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)

        output = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(
                iam,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False)
            output['pred_iam'] = iam

        return output


@MODELS.register_module()
class DRIAMDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 ins_dim=256,
                 ins_conv=4,
                 mask_dim=256,
                 mask_conv=4,
                 kernel_dim=128,
                 scale_factor=2.0,
                 style='lp',
                 groups=4,
                 dyscope=False,
                 insert_ind=(1, 3),
                 output_iam=False,
                 num_masks=100,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        # coordconv: add 2 for coordinates
        in_channels = in_channels   # ENCODER.NUM_CHANNELS + 2

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(
            in_channels,
            dim=ins_dim,
            num_convs=ins_conv,
            num_masks=num_masks,
            num_classes=num_classes,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)
        self.drmask_branch = DRMaskBranch(
            in_channels,
            dim=mask_dim,
            num_convs=mask_conv,
            kernel_dim=kernel_dim,
            scale=scale_factor,
            style=style,
            groups=groups,
            dyscope=dyscope,
            insert_ind=insert_ind,
            act_cfg=act_cfg)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        mask_features = self.drmask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel,
                               mask_features.view(B, C,
                                                  H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)

        output = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(
                iam,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False)
            output['pred_iam'] = iam

        return output