import torch.nn as nn
import torch.nn.functional as F
import torch

class ReceptiveFieldEnhancement(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        # Initialize weights
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        min_h = min(x1.size(2), x2.size(2), x3.size(2))
        min_w = min(x1.size(3), x2.size(3), x3.size(3))

        x1 = F.interpolate(x1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(min_h, min_w), mode='bilinear', align_corners=False)

        out = self.norm(x1 + x2 + x3)
        return self.activation(out)