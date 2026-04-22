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


class TinyDenoiseBranch(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.to_rgb = nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x + self.block(x)
        return self.to_rgb(feat)


class GateBranch(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LMFANetAdaptiveDenoise(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()

        # base LMFA backbone preserved
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

        # dehaze head (same role as original final conv, just 3x3 after reduced features)
        self.dehaze_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1, bias=True)

        # new tiny branches
        self.denoise_branch = TinyDenoiseBranch(base_channels)
        self.gate_branch = GateBranch(base_channels)

        self.sigmoid = nn.Sigmoid()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        fused = torch.cat([f1, f2, f3], dim=1)
        reduced = self.maxout(fused)
        return reduced

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        feat = self.extract_features(x)

        dehaze_out = self.sigmoid(self.dehaze_head(feat))
        denoise_out = self.sigmoid(self.denoise_branch(feat))
        gate = self.gate_branch(feat)   # B,1,H,W

        fused_out = (1.0 - gate) * dehaze_out + gate * denoise_out

        if return_gate:
            return fused_out, gate, dehaze_out, denoise_out
        return fused_out
