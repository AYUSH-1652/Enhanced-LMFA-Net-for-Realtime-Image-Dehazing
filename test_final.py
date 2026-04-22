import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(".")

from models.lmfanet import LMFANetAdaptiveDenoiseSE
from utils.metrics import compute_psnr_ssim


def main():
    TEST_HAZY = "/kaggle/input/datasets/balraj98/synthetic-objective-testing-set-sots-reside/outdoor/hazy"
    TEST_CLEAR = "/kaggle/input/datasets/balraj98/synthetic-objective-testing-set-sots-reside/outdoor/clear"

    CKPT_PATH = "/kaggle/working/checkpoints_final/best.pth"
    SAVE_DIR = "/kaggle/working/sots_final_outputs"

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LMFANetAdaptiveDenoiseSE().to(device)
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
            dehaze_out = dehaze_out.squeeze(0).cpu().clamp(0, 1)
            denoise_out = denoise_out.squeeze(0).cpu().clamp(0, 1)
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

                fig = plt.figure(figsize=(20, 4))

                plt.subplot(1, 5, 1)
                plt.title("Input")
                plt.imshow(hazy_img)
                plt.axis("off")

                plt.subplot(1, 5, 2)
                plt.title("Dehaze Head")
                plt.imshow(to_pil(dehaze_out))
                plt.axis("off")

                plt.subplot(1, 5, 3)
                plt.title("Denoise Head")
                plt.imshow(to_pil(denoise_out))
                plt.axis("off")

                plt.subplot(1, 5, 4)
                plt.title("Final Output")
                plt.imshow(to_pil(pred))
                plt.axis("off")

                plt.subplot(1, 5, 5)
                plt.title("Ground Truth")
                plt.imshow(clear_vis)
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR, f"result_{saved_visuals+1}_{fname}.png"), bbox_inches="tight")
                plt.show()
                plt.close(fig)

                gate_fig = plt.figure(figsize=(5, 4))
                plt.title("Gate Map")
                plt.imshow(gate_map, cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR, f"gate_{saved_visuals+1}_{fname}.png"), bbox_inches="tight")
                plt.show()
                plt.close(gate_fig)

                saved_visuals += 1

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(hazy_files)}")

    avg_psnr = sum(all_psnr) / len(all_psnr)
    avg_ssim = sum(all_ssim) / len(all_ssim)

    print("\nFinal Model SOTS Outdoor Results")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Saved 5 visuals to: {SAVE_DIR}")


if __name__ == "__main__":
    main()