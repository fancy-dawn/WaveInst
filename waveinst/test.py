import torch
import torch.nn as nn

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

# Adaptive Gated Fusion Module (AGFM)
class AGFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.se_fpn = SE_Block(in_channels1)
        self.se_dwt = SE_Block(in_channels2)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1, bias=False),
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

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试代码
if __name__ == "__main__":
    # 创建输入张量 (batch_size=1, channels=256, height=64, width=64)
    feat_fpn = torch.randn(1, 256, 64, 64)
    feat_dwt = torch.randn(1, 256, 64, 64)

    # 创建 AGFM 模块
    agfm = AGFM(in_channels1=256, in_channels2=256, out_channels=256)

    # 计算参数量
    total_params = count_parameters(agfm)
    print(f"AGFM 参数量: {total_params:,}")

    # 计算前向传播输出
    fused_feature = agfm(feat_fpn, feat_dwt)
    print("输出形状:", fused_feature.shape)
