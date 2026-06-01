"""
Compact U-Net for built-up / impervious-surface segmentation of Sentinel-2 tiles.

This is the model that replaces the v1 NDBI threshold. Input is a multi-band
image tile (C, H, W); output is a per-pixel logit for "built-up". Standard
3-level U-Net — small enough to read, real enough to train on a GPU at scale.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """3-level U-Net. Input H, W must be divisible by 8."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))
        x = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), c2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), c1], dim=1))
        return self.head(x)


if __name__ == "__main__":
    m = UNet(in_channels=4)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 4, 64, 64)
    print(f"UNet params: {n_params:,}  | output shape: {tuple(m(x).shape)}")
