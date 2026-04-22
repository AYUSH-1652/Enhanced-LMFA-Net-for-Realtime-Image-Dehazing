import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import sys
sys.path.append("/kaggle/working")

from models.lmfanet import LMFANet

# =========================
# 1. Paths
# =========================
SOTS_HAZY = r"data\RESIDE\SOTS\outdoor\hazy"
SOTS_CLEAR = r"data\RESIDE\SOTS\outdoor\clear"
CKPT_PATH = r"checkpoints\baseline\best.pth"
SAVE_DIR = r"outputs\sots_baseline"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 2. Helpers
# =========================
def get_clear_name(hazy_name: str) -> str:
    base, _ = os.path.splitext(hazy_name)
    clear_base = base.split("_")[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = clear_base + ext
        if os.path.exists(os.path.join(SOTS_CLEAR, candidate)):
            return candidate
    raise FileNotFoundError(f"No clear pair found for {hazy_name}")

def tensor_to_numpy_img(tensor):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img.clip(0.0, 1.0)
    return img

def compute_psnr_ssim(pred, target):
    pred_np = tensor_to_numpy_img(pred)
    target_np = tensor_to_numpy_img(target)

    pred_uint8 = (pred_np * 255.0).round().astype("uint8")
    target_uint8 = (target_np * 255.0).round().astype("uint8")

    psnr = peak_signal_noise_ratio(target_uint8, pred_uint8, data_range=255)
    ssim = structural_similarity(target_uint8, pred_uint8, channel_axis=2, data_range=255)
    return psnr, ssim

# =========================
# 3. Load model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LMFANet().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Loaded checkpoint:", CKPT_PATH)

# =========================
# 4. Test loop
# =========================
transform = T.ToTensor()

hazy_files = sorted([
    f for f in os.listdir(SOTS_HAZY)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

all_psnr = []
all_ssim = []

num_visuals_to_save = 5
saved_visuals = 0

with torch.no_grad():
    for idx, hazy_name in enumerate(hazy_files):
        clear_name = get_clear_name(hazy_name)

        hazy_img = Image.open(os.path.join(SOTS_HAZY, hazy_name)).convert("RGB")
        clear_img = Image.open(os.path.join(SOTS_CLEAR, clear_name)).convert("RGB")

        hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
        output = model(hazy_tensor)

        # LMFA-Net uses valid convs, so output is smaller by 10 pixels total
        clear_tensor = transform(clear_img)
        clear_tensor = clear_tensor[:, 5:-5, 5:-5]

        pred = output.squeeze(0).cpu().clamp(0, 1)

        if pred.shape != clear_tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {hazy_name}: pred={pred.shape}, target={clear_tensor.shape}"
            )

        psnr, ssim = compute_psnr_ssim(pred, clear_tensor)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        if saved_visuals < num_visuals_to_save:
            pred_pil = T.ToPILImage()(pred)
            target_pil = T.ToPILImage()(clear_tensor)

            fig = plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Hazy Input")
            plt.imshow(hazy_img)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Dehazed Output")
            plt.imshow(pred_pil)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Clear Target")
            plt.imshow(target_pil)
            plt.axis("off")

            save_path = os.path.join(SAVE_DIR, f"result_{saved_visuals+1}_{hazy_name}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

            saved_visuals += 1

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(hazy_files)} images")

# =========================
# 5. Final metrics
# =========================
avg_psnr = sum(all_psnr) / len(all_psnr)
avg_ssim = sum(all_ssim) / len(all_ssim)

print(f"\nSOTS Outdoor Results")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Saved {saved_visuals} visual comparison images to: {SAVE_DIR}")