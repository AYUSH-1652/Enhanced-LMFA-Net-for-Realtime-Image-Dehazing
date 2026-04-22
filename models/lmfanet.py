import torch
import torch.nn as nn


class MaxoutReduce(nn.Module):
    def __init__(self, in_channels: int = 48, out_channels: int = 16, pieces: int = 3):
        super().__init__()
        if in_channels != out_channels * pieces:
            raise ValueError("in_channels must equal out_channels * pieces")
        self.out_channels = out_channels
        self.pieces = pieces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, self.out_channels, self.pieces, h, w)
        x, _ = torch.max(x, dim=2)
        return x


class LMFANet(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()

        self.branch1 = nn.Conv2d(3, base_channels, kernel_size=7, padding=0, bias=True)

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=5, padding=0, bias=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=0, bias=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=0, bias=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=0, bias=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=0, bias=True),
        )

        self.maxout = MaxoutReduce(in_channels=48, out_channels=16, pieces=3)
        self.final = nn.Conv2d(base_channels, 3, kernel_size=5, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        fused = torch.cat([f1, f2, f3], dim=1)
        reduced = self.maxout(fused)

        out = self.final(reduced)
        out = self.sigmoid(out)
        return out
