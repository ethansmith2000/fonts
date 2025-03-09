import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            downsample=False, 
            upsample=False, 
            # norm_type="batch"
            norm_type="group",
            norm_groups=32,
            ):
        super(ResBlock, self).__init__()
        
        stride = 2 if downsample else 1
        self.upsample = upsample

        if norm_type == "batch":
            norm_fn = lambda channels: nn.BatchNorm2d(channels)
        elif norm_type == "group":
            norm_fn = lambda channels: nn.GroupNorm(num_groups=norm_groups, num_channels=channels)

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = norm_fn(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = norm_fn(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                norm_fn(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = F.interpolate(identity, scale_factor=2, mode='bilinear', align_corners=False)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu2(out)
        
        return out
