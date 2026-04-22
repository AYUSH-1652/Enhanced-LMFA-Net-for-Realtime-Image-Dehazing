import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyiqa


# =========================================================
# BASELINE MODEL
# =========================================================
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


class LMFANetBaseline(nn.Module):
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


# =========================================================
# ADAPTIVE DENOISE MODEL
# =========================================================
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

        self.dehaze_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1, bias=True)
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
        gate = self.gate_branch(feat)

        fused_out = (1.0 - gate) * dehaze_out + gate * denoise_out

        if return_gate:
            return fused_out, gate, dehaze_out, denoise_out
        return fused_out


# =========================================================
# FINAL MODEL = ADAPTIVE DENOISE + SE ATTENTION
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LMFANetAdaptiveDenoiseSE(nn.Module):
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

        self.se = SEBlock(channels=48, reduction=8)
        self.maxout = MaxoutReduce(in_channels=48, out_channels=16, pieces=3)

        self.dehaze_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1, bias=True)
        self.denoise_branch = TinyDenoiseBranch(base_channels)
        self.gate_branch = GateBranch(base_channels)

        self.sigmoid = nn.Sigmoid()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        fused = torch.cat([f1, f2, f3], dim=1)
        fused = self.se(fused)
        reduced = self.maxout(fused)
        return reduced

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        feat = self.extract_features(x)

        dehaze_out = self.sigmoid(self.dehaze_head(feat))
        denoise_out = self.sigmoid(self.denoise_branch(feat))
        gate = self.gate_branch(feat)

        fused_out = (1.0 - gate) * dehaze_out + gate * denoise_out

        if return_gate:
            return fused_out, gate, dehaze_out, denoise_out
        return fused_out


# =========================================================
# HELPERS
# =========================================================
def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)


def preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)


def tensor_to_bgr(pred_tensor: torch.Tensor) -> np.ndarray:
    pred = pred_tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    pred = (pred * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)


def crop_original_for_lmfa(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    return img_bgr[5:h - 5, 5:w - 5]


def compute_entropy(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray, bins=256, range=(0, 256))[0].astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def compute_laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def avg(x):
    return float(np.mean(x)) if len(x) > 0 else float("nan")


def evaluate_metrics(img_bgr: np.ndarray, niqe_metric, brisque_metric, piqe_metric, device):
    tensor_metric = preprocess_bgr(img_bgr).to(device)

    niqe = float(niqe_metric(tensor_metric).item())
    brisque = float(brisque_metric(tensor_metric).item())
    piqe = float(piqe_metric(tensor_metric).item())
    entropy = compute_entropy(img_bgr)
    lap_var = compute_laplacian_variance(img_bgr)

    return {
        "niqe": niqe,
        "brisque": brisque,
        "piqe": piqe,
        "entropy": entropy,
        "lap_var": lap_var,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs adaptive vs final model on RTTS")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="checkpoints/baseline/best.pth",
        help="Path to baseline checkpoint"
    )
    parser.add_argument(
        "--adaptive_model",
        type=str,
        default="checkpoints/adaptive_denoise/best.pth",
        help="Path to adaptive denoise checkpoint"
    )
    parser.add_argument(
        "--final_model",
        type=str,
        default="checkpoints/attentionnet_edgeloss_adaptive/best.pth",
        help="Path to final checkpoint"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="RTTS/JPEGImages",
        help="RTTS image folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/rtts_compare_all3",
        help="Directory to save visuals"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="outputs/rtts_compare_all3/results_all3.csv",
        help="CSV output path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=5,
        help="Number of comparison visuals to save"
    )
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)

    baseline_model = LMFANetBaseline().to(device)
    adaptive_model = LMFANetAdaptiveDenoise().to(device)
    final_model = LMFANetAdaptiveDenoiseSE().to(device)

    load_checkpoint(baseline_model, args.baseline_model, device)
    load_checkpoint(adaptive_model, args.adaptive_model, device)
    load_checkpoint(final_model, args.final_model, device)

    baseline_model.eval()
    adaptive_model.eval()
    final_model.eval()

    print("Loaded all 3 checkpoints successfully")

    niqe_metric = pyiqa.create_metric("niqe", device=device)
    brisque_metric = pyiqa.create_metric("brisque", device=device)
    piqe_metric = pyiqa.create_metric("piqe", device=device)

    print("Loaded NIQE, BRISQUE, PIQE")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = sorted([
        str(p) for p in Path(args.input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    if not image_paths:
        raise RuntimeError(f"No valid images found in: {args.input_dir}")

    rows = []
    saved = 0

    input_scores = {"niqe": [], "brisque": [], "piqe": [], "entropy": [], "lap_var": []}
    baseline_scores = {"niqe": [], "brisque": [], "piqe": [], "entropy": [], "lap_var": []}
    adaptive_scores = {"niqe": [], "brisque": [], "piqe": [], "entropy": [], "lap_var": [], "gate": []}
    final_scores = {"niqe": [], "brisque": [], "piqe": [], "entropy": [], "lap_var": [], "gate": []}

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths, start=1):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            h, w = img_bgr.shape[:2]
            if h <= 10 or w <= 10:
                print(f"Skipping too-small image: {img_path}")
                continue

            inp_tensor = preprocess_bgr(img_bgr).to(device)

            baseline_pred = baseline_model(inp_tensor)
            baseline_bgr = tensor_to_bgr(baseline_pred)

            adaptive_pred, adaptive_gate, _, _ = adaptive_model(inp_tensor, return_gate=True)
            adaptive_bgr = tensor_to_bgr(adaptive_pred)
            adaptive_gate_mean = float(adaptive_gate.mean().item())

            final_pred, final_gate, _, _ = final_model(inp_tensor, return_gate=True)
            final_bgr = tensor_to_bgr(final_pred)
            final_gate_mean = float(final_gate.mean().item())

            input_cropped = crop_original_for_lmfa(img_bgr)

            input_m = evaluate_metrics(input_cropped, niqe_metric, brisque_metric, piqe_metric, device)
            baseline_m = evaluate_metrics(baseline_bgr, niqe_metric, brisque_metric, piqe_metric, device)
            adaptive_m = evaluate_metrics(adaptive_bgr, niqe_metric, brisque_metric, piqe_metric, device)
            final_m = evaluate_metrics(final_bgr, niqe_metric, brisque_metric, piqe_metric, device)

            for k in input_scores:
                input_scores[k].append(input_m[k])
                baseline_scores[k].append(baseline_m[k])
                adaptive_scores[k].append(adaptive_m[k])
                final_scores[k].append(final_m[k])

            adaptive_scores["gate"].append(adaptive_gate_mean)
            final_scores["gate"].append(final_gate_mean)

            rows.append({
                "image": os.path.basename(img_path),

                "input_niqe": input_m["niqe"],
                "baseline_niqe": baseline_m["niqe"],
                "adaptive_niqe": adaptive_m["niqe"],
                "final_niqe": final_m["niqe"],

                "input_brisque": input_m["brisque"],
                "baseline_brisque": baseline_m["brisque"],
                "adaptive_brisque": adaptive_m["brisque"],
                "final_brisque": final_m["brisque"],

                "input_piqe": input_m["piqe"],
                "baseline_piqe": baseline_m["piqe"],
                "adaptive_piqe": adaptive_m["piqe"],
                "final_piqe": final_m["piqe"],

                "input_entropy": input_m["entropy"],
                "baseline_entropy": baseline_m["entropy"],
                "adaptive_entropy": adaptive_m["entropy"],
                "final_entropy": final_m["entropy"],

                "input_lap_var": input_m["lap_var"],
                "baseline_lap_var": baseline_m["lap_var"],
                "adaptive_lap_var": adaptive_m["lap_var"],
                "final_lap_var": final_m["lap_var"],

                "adaptive_gate_mean": adaptive_gate_mean,
                "final_gate_mean": final_gate_mean,
            })

            if saved < args.num_visuals:
                side = np.hstack([input_cropped, baseline_bgr, adaptive_bgr, final_bgr])
                save_path = output_dir / f"compare_{saved+1}_{Path(img_path).name}"
                cv2.imwrite(str(save_path), side)
                saved += 1

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(image_paths)} images")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_path, index=False)

    print("\n===== RTTS BASELINE VS ADAPTIVE VS FINAL RESULTS =====")
    print(f"Total processed images: {len(df)}")

    print("\n--- INPUT AVERAGES ---")
    print(f"NIQE      : {avg(input_scores['niqe']):.4f}")
    print(f"BRISQUE   : {avg(input_scores['brisque']):.4f}")
    print(f"PIQE      : {avg(input_scores['piqe']):.4f}")
    print(f"Entropy   : {avg(input_scores['entropy']):.4f}")
    print(f"Lap Var   : {avg(input_scores['lap_var']):.4f}")

    print("\n--- BASELINE AVERAGES ---")
    print(f"NIQE      : {avg(baseline_scores['niqe']):.4f}")
    print(f"BRISQUE   : {avg(baseline_scores['brisque']):.4f}")
    print(f"PIQE      : {avg(baseline_scores['piqe']):.4f}")
    print(f"Entropy   : {avg(baseline_scores['entropy']):.4f}")
    print(f"Lap Var   : {avg(baseline_scores['lap_var']):.4f}")

    print("\n--- ADAPTIVE AVERAGES ---")
    print(f"NIQE      : {avg(adaptive_scores['niqe']):.4f}")
    print(f"BRISQUE   : {avg(adaptive_scores['brisque']):.4f}")
    print(f"PIQE      : {avg(adaptive_scores['piqe']):.4f}")
    print(f"Entropy   : {avg(adaptive_scores['entropy']):.4f}")
    print(f"Lap Var   : {avg(adaptive_scores['lap_var']):.4f}")
    print(f"Gate Mean : {avg(adaptive_scores['gate']):.4f}")

    print("\n--- FINAL AVERAGES ---")
    print(f"NIQE      : {avg(final_scores['niqe']):.4f}")
    print(f"BRISQUE   : {avg(final_scores['brisque']):.4f}")
    print(f"PIQE      : {avg(final_scores['piqe']):.4f}")
    print(f"Entropy   : {avg(final_scores['entropy']):.4f}")
    print(f"Lap Var   : {avg(final_scores['lap_var']):.4f}")
    print(f"Gate Mean : {avg(final_scores['gate']):.4f}")

    print("\nLower is better: NIQE, BRISQUE, PIQE")
    print("Higher is generally better: Entropy, Laplacian Variance")

    print(f"\nCSV saved to: {args.csv_path}")
    print(f"Comparison visuals saved to: {args.output_dir}")


if __name__ == "__main__":
    main()