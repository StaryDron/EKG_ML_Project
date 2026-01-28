import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_bottleneck = self.bottleneck(x)
        out_convs = [c(out_bottleneck) for c in self.convs]
        out_pool = self.maxpool_conv(x)
        out = torch.cat(out_convs + [out_pool], dim=1)
        return self.relu(self.bn(out))


class InceptionTime1D(nn.Module):
    def __init__(self, in_channels=12, n_classes=5, num_modules=6, channels=64):
        super().__init__()
        self.modules_list = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_modules):
            self.modules_list.append(InceptionModule(current_channels, channels))
            current_channels = channels

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        for mod in self.modules_list:
            x = mod(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)