import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("/kaggle/working")

from models.lmfanet_adaptive_denoise import LMFANetAdaptiveDenoise
from utils.metrics import compute_psnr_ssim

TEST_HAZY = r"data\RESIDE\SOTS\outdoor\hazy"
TEST_CLEAR = r"data\RESIDE\SOTS\outdoor\clear"

CKPT_PATH = r"checkpoints\adaptive_denoise\best.pth"
SAVE_DIR = r"outputs\sots_adaptive_denoise"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LMFANetAdaptiveDenoise().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

hazy_files = sorted([
    f for f in os.listdir(TEST_HAZY)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

all_psnr = []
all_ssim = []
saved_visuals = 0

with torch.no_grad():
    for idx, fname in enumerate(hazy_files):
        clear_name = fname.split("_")[0] + ".png"
        if not os.path.exists(os.path.join(TEST_CLEAR, clear_name)):
            clear_name = fname.split("_")[0] + ".jpg"

        hazy_path = os.path.join(TEST_HAZY, fname)
        clear_path = os.path.join(TEST_CLEAR, clear_name)

        if not os.path.exists(clear_path):
            print(f"Skipping {fname} because GT not found")
            continue

        hazy_img = Image.open(hazy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")

        inp = to_tensor(hazy_img).unsqueeze(0).to(device)
        pred, gate, dehaze_out, denoise_out = model(inp, return_gate=True)

        pred = pred.squeeze(0).cpu().clamp(0, 1)
        clear = to_tensor(clear_img)

        _, ph, pw = pred.shape
        _, ch, cw = clear.shape

        if ch < ph or cw < pw:
            raise ValueError(f"Clear image smaller than prediction: clear={clear.shape}, pred={pred.shape}")

        top = (ch - ph) // 2
        left = (cw - pw) // 2
        clear = clear[:, top:top + ph, left:left + pw]

        psnr, ssim = compute_psnr_ssim(pred, clear)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        if saved_visuals < 5:
            gate_map = gate.squeeze(0).squeeze(0).cpu().numpy()
            clear_vis = to_pil(clear)

            fig = plt.figure(figsize=(16, 4))

            plt.subplot(1, 4, 1)
            plt.title("Input")
            plt.imshow(hazy_img)
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.title("Output")
            plt.imshow(to_pil(pred))
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.title("Ground Truth")
            plt.imshow(clear_vis)
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.title("Gate")
            plt.imshow(gate_map, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"result_{saved_visuals+1}_{fname}.png"), bbox_inches="tight")
            plt.show()
            plt.close(fig)
            saved_visuals += 1

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(hazy_files)}")

avg_psnr = sum(all_psnr) / len(all_psnr)
avg_ssim = sum(all_ssim) / len(all_ssim)

print("\nSOTS Outdoor Results")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Saved 5 visuals to: {SAVE_DIR}")