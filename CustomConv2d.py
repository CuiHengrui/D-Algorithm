import torch.nn as nn
import torch.nn.functional as F

# 自定义BasicConv2d
class CustomBasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, track_running_stats=False)  # 关键参数

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)