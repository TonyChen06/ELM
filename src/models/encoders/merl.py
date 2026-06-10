"""MERL signal encoder: 1-D ResNet-101 trunk pooled to num_tokens tokens."""
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout * 4, 1, bias=False), nn.BatchNorm1d(cout * 4),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout * 4:
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout * 4, 1, stride=stride, bias=False),
                                          nn.BatchNorm1d(cout * 4))

    def forward(self, x):
        return torch.relu(self.net(x) + self.shortcut(x))


class Merl(nn.Module):
    def __init__(self, num_tokens, layers=(3, 4, 23, 3)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        stages, cin = [], 64
        for i, (cout, n) in enumerate(zip((64, 128, 256, 512), layers)):
            for j in range(n):
                stages.append(Bottleneck(cin, cout, stride=2 if (i and j == 0) else 1))
                cin = cout * 4
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, ecg):
        x = self.stages(self.stem(ecg.float()))
        return self.pool(x).transpose(1, 2)  # (B, num_tokens, 2048)
