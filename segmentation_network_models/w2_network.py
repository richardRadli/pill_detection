import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(SimpleUNet, self).__init__()
        self.enc1 = UNetBlock(in_channels, 16)
        self.enc2 = UNetBlock(16, 32)
        self.enc3 = UNetBlock(32, 64)
        self.enc4 = UNetBlock(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        self.dec4 = UNetBlock(128 + 64, 64)
        self.dec3 = UNetBlock(64 + 32, 32)
        self.dec2 = UNetBlock(32 + 16, 16)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder path with upsampling and concatenation
        d4 = self.dec4(torch.cat([F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))

        out = self.final_conv(d2)
        return out


class W2Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(W2Net, self).__init__()
        # Define four U-Net instances with fixed intermediate output channels
        self.unet1 = SimpleUNet(in_channels, 16)
        self.unet2 = SimpleUNet(16, 16)
        self.unet3 = SimpleUNet(16, 16)
        self.unet4 = SimpleUNet(16, out_channels)  # Final U-Net outputs desired output channels

    def forward(self, x):
        # Pass input through each U-Net sequentially
        u1_out = self.unet1(x)
        u2_out = self.unet2(u1_out)
        u3_out = self.unet3(u2_out)
        u4_out = self.unet4(u3_out)

        return u4_out

