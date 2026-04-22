import os
import random
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF


def default_image_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class RESIDEPairedDataset(Dataset):
    def __init__(self, hazy_dir: str, clear_dir: str, crop_size: int | None = None, training: bool = True):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.crop_size = crop_size
        self.training = training

        self.border = 5
        self.shrink = 10

        if crop_size is not None and crop_size <= self.shrink:
            raise ValueError(f"crop_size must be greater than {self.shrink}, got {crop_size}")

        if not os.path.isdir(hazy_dir):
            raise FileNotFoundError(f"Hazy directory not found: {hazy_dir}")
        if not os.path.isdir(clear_dir):
            raise FileNotFoundError(f"Clear directory not found: {clear_dir}")

        self.hazy_files = sorted(
            [f for f in os.listdir(hazy_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        )
        if not self.hazy_files:
            raise RuntimeError(f"No images found in hazy directory: {hazy_dir}")

        self.to_tensor = T.ToTensor()
        self.pairs = []

        for hazy_name in self.hazy_files:
            clear_name = self._get_clear_name(hazy_name)
            self.pairs.append((hazy_name, clear_name))

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_clear_name(self, hazy_name: str) -> str:
        base, _ = os.path.splitext(hazy_name)

        direct = os.path.join(self.clear_dir, hazy_name)
        if os.path.exists(direct):
            return hazy_name

        clear_base = base.split("_")[0]
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = clear_base + ext
            if os.path.exists(os.path.join(self.clear_dir, candidate)):
                return candidate

        raise FileNotFoundError(f"Could not find matching clear image for: {hazy_name}")

    def _paired_random_crop(self, hazy: Image.Image, clear: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.crop_size is None:
            return hazy, clear

        if hazy.width != clear.width or hazy.height != clear.height:
            raise ValueError("Hazy and clear images must have the same original size.")

        if hazy.width < self.crop_size or hazy.height < self.crop_size:
            raise ValueError(f"Crop size {self.crop_size} is larger than image size {hazy.size}")

        i, j, h, w = T.RandomCrop.get_params(hazy, output_size=(self.crop_size, self.crop_size))
        hazy_crop = TF.crop(hazy, i, j, h, w)
        clear_crop = TF.crop(clear, i + self.border, j + self.border, h - self.shrink, w - self.shrink)

        return hazy_crop, clear_crop

    def _center_crop_clear_for_full_image(self, clear: Image.Image) -> Image.Image:
        if clear.width <= self.shrink or clear.height <= self.shrink:
            raise ValueError(f"Image too small for valid LMFA-Net alignment: {clear.size}")

        return TF.crop(clear, self.border, self.border, clear.height - self.shrink, clear.width - self.shrink)

    def __getitem__(self, idx: int):
        hazy_name, clear_name = self.pairs[idx]

        hazy = default_image_loader(os.path.join(self.hazy_dir, hazy_name))
        clear = default_image_loader(os.path.join(self.clear_dir, clear_name))

        if hazy.size != clear.size:
            raise ValueError(
                f"Size mismatch for pair: hazy={hazy_name} ({hazy.size}) vs clear={clear_name} ({clear.size})"
            )

        if self.training and self.crop_size is not None:
            hazy, clear = self._paired_random_crop(hazy, clear)

            if random.random() > 0.5:
                hazy = TF.hflip(hazy)
                clear = TF.hflip(clear)
        else:
            clear = self._center_crop_clear_for_full_image(clear)

        hazy = self.to_tensor(hazy)
        clear = self.to_tensor(clear)

        return hazy, clear, hazy_name
