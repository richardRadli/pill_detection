""" Full assembly of the parts to form the complete network """

from torch.utils.checkpoint import checkpoint

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64//4))
        self.down1 = (Down(64//4, 128//4))
        self.down2 = (Down(128//4, 256//4))
        self.down3 = (Down(256//4, 512//4))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512//4, 1024//4 // factor))
        self.up1 = (Up(1024//4, 512//4 // factor, bilinear))
        self.up2 = (Up(512//4, 256//4 // factor, bilinear))
        self.up3 = (Up(256//4, 128//4 // factor, bilinear))
        self.up4 = (Up(128//4, 64//4, bilinear))
        self.outc = (OutConv(64//4, n_classes))

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
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = checkpoint(self.inc)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.outc = checkpoint(self.outc)


class StackedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(StackedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Define the first UNet
        self.unet1 = UNet(n_channels, n_classes, bilinear=bilinear)

        # Define the second UNet
        self.unet2 = UNet(n_channels + n_classes, n_classes, bilinear=bilinear)

        # Define the third UNet
        self.unet3 = UNet(n_channels + 2*n_classes, n_classes, bilinear=bilinear)

    def forward(self, x):
        # Pass input through the first UNet
        x1 = self.unet1(x)

        # Concatenate output of first UNet with input and pass through the second UNet
        x2_input = torch.cat([x, func.interpolate(x1, scale_factor=1)], dim=1)
        x2 = self.unet2(x2_input)

        # Concatenate output of second UNet with input and pass through the third UNet
        x3_input = torch.cat([x, func.interpolate(x1, scale_factor=1), func.interpolate(x2, scale_factor=1)], dim=1)
        x3 = self.unet3(x3_input)

        # Return the output of the third UNet
        return x3

    def use_checkpointing(self):
        self.unet1.use_checkpointing()
        self.unet2.use_checkpointing()
        self.unet3.use_checkpointing()
