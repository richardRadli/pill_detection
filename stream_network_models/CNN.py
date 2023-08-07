"""
File: cnn.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program implements the CNN.
"""

import torch
import torch.nn as nn


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++ S T R E A M   N E T +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CNN(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, loc: list) -> None:
        """
        Args:
            loc: list of number of channels
        Returns:
            None
        """
        
        super(CNN, self).__init__()

        self.loc = loc
        self.conv1 = nn.Conv2d(loc[0], loc[1], kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(loc[1], loc[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(loc[1], loc[1], kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(loc[1], loc[2], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(loc[2], loc[2], kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(loc[2], loc[2], kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv7 = nn.Conv2d(loc[2], loc[3], kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(loc[3], loc[3], kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(loc[3], loc[3], kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = nn.Conv2d(loc[3], loc[4], kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(loc[4], loc[4], kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(loc[4], loc[4], kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv13 = nn.Conv2d(loc[4], loc[5], kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(loc[5], loc[5], kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(loc[5], loc[5], kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(loc[5] * 4 * 4, loc[6])
        self.reg = nn.Linear(loc[6], loc[4])

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the forward function of the FusionNet.

        Args:
            x (torch.Tensor): input tensor.
        Returns:
            x (torch.Tensor): output tensor.
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool3(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool4(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = x.view(-1, self.loc[5] * 4 * 4)
        x = self.fc(x)
        x = self.reg(x)

        return x
