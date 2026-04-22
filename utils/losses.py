import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        xs = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-((xs - window_size // 2) ** 2) / (2 * sigma**2))
        return gauss / gauss.sum()

    def create_window(self, window_size: int, channel: int, device: torch.device) -> torch.Tensor:
        one_d = self.gaussian(window_size, 1.5).unsqueeze(1)
        two_d = one_d @ one_d.t()
        window = two_d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        channel = img1.size(1)
        window = self.create_window(self.window_size, channel, img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim(pred, target)


class CombinedLoss(nn.Module):
    def __init__(self, alpha_ssim: float = 0.02):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.alpha_ssim = alpha_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l2 = self.mse(pred, target)
        ls = self.ssim_loss(pred, target)
        return l2 + self.alpha_ssim * ls
