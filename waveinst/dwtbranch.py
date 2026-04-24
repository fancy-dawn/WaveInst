import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from mmdet.registry import MODELS

class DWTBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mode='zero',
                 wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, mode=mode, wave=wave)
        self.convH = nn.Conv2d(in_channels * 3, out_channels // 4 * 3, kernel_size=3, padding=1)
        self.convL = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)

        self.bnH = nn.BatchNorm2d(out_channels // 4 * 3)
        self.bnL = nn.BatchNorm2d(out_channels // 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        xL, xH = self.dwt(x)
        x_HL = xH[0][:, :, 0, ::]
        x_LH = xH[0][:, :, 1, ::]
        x_HH = xH[0][:, :, 2, ::]
        xH = torch.cat([x_HL, x_LH, x_HH], dim=1)

        xL = self.relu(self.bnL(self.convL(xL)))
        xH = self.relu(self.bnH(self.convH(xH)))

        return xL, xH
    

class HFEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 3 == 0
        channels = in_channels // 3
        self.path1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.path2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.path3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        path1 = self.path1(x1)
        path2 = self.path2(x2)
        path3 = self.path3(x3)
        x_cat = torch.cat([path1, path2, path3], dim=1)
        x_cat = self.relu(self.bn(self.conv(x_cat)))
        return x_cat + x


@MODELS.register_module()
class DWTBranch(nn.Module):
    def __init__(self,
                 channels=16,
                 mode='zero',
                 wave='haar',
                 with_l=False):
        super().__init__()
        print(channels)
        c = channels
        self.with_l = with_l
        self.conv = nn.Conv2d(3, c, kernel_size=1)
        self.dwt0 = DWTBlock(c, 4*c, mode=mode, wave=wave)
        self.down00 = nn.Conv2d(3*c, 3*c, kernel_size=3, stride=2, padding=1)
        self.hfe00 = HFEBlock(3*c)
        self.down01 = nn.Conv2d(3*c, 3*c, kernel_size=3, stride=2, padding=1)
        self.hfe01 = HFEBlock(3*c)
        self.dwt1 = DWTBlock(c, 4*c)
        self.down10 = nn.Conv2d(3*c, 3*c, kernel_size=3, stride=2, padding=1)
        self.hfe10 = HFEBlock(3*c)
        self.dwt2 = DWTBlock(c ,4*c, mode=mode, wave=wave)
        self.hfe20 = HFEBlock(3*c)
        if with_l:
            self.fusion = nn.Conv2d(10*c, 256, kernel_size=1)
        else:
            self.fusion = nn.Conv2d(9*c, 256, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        xL0, xH0 = self.dwt0(x)
        xH0 = self.down00(xH0)
        xH0 = self.hfe00(xH0)
        xH0 = self.down01(xH0)
        xH0 = self.hfe01(xH0)

        xL1, xH1 = self.dwt1(xL0)
        xH1 = self.down10(xH1)
        xH1 = self.hfe10(xH1)

        xL2, xH2 = self.dwt2(xL1)
        xH2 = self.hfe20(xH2)

        if self.with_l:
            x = torch.cat([xL2, xH0, xH1, xH2], dim=1)
            return self.fusion(x)
        else:
            x = torch.cat([xH0, xH1, xH2], dim=1)
            return self.fusion(x)