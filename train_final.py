import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lmfanet import LMFANetAdaptiveDenoiseSE
from utils.dataset import RESIDEPairedDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_psnr_ssim


def center_crop_to_match(pred, ref):
    if pred.shape[2:] == ref.shape[2:]:
        return pred

    dh = pred.shape[2] - ref.shape[2]
    dw = pred.shape[3] - ref.shape[3]

    if dh < 0 or dw < 0:
        raise RuntimeError(
            f"Prediction smaller than reference: pred={pred.shape}, ref={ref.shape}"
        )

    top = dh // 2
    left = dw // 2
    return pred[:, :, top:top + ref.shape[2], left:left + ref.shape[3]]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_gate = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for hazy, clear, _ in pbar:
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred, gate, _, _ = model(hazy, return_gate=True)

        if pred.shape != clear.shape:
            pred = center_crop_to_match(pred, clear)
            gate = center_crop_to_match(gate, clear)

        loss = criterion(pred, clear)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gate += gate.mean().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            gate=f"{gate.mean().item():.4f}"
        )

    avg_loss = total_loss / len(loader)
    avg_gate = total_gate / len(loader)
    return avg_loss, avg_gate


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_gate = 0.0
    count = 0

    for hazy, clear, _ in tqdm(loader, desc="Validation", leave=False):
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        pred, gate, _, _ = model(hazy, return_gate=True)

        if pred.shape != clear.shape:
            pred = center_crop_to_match(pred, clear)
            gate = center_crop_to_match(gate, clear)

        loss = criterion(pred, clear)
        total_loss += loss.item()
        total_gate += gate.mean().item()

        for b in range(pred.size(0)):
            psnr, ssim = compute_psnr_ssim(pred[b], clear[b])
            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_gate = total_gate / len(loader)

    return avg_loss, avg_psnr, avg_ssim, avg_gate


def save_checkpoint(path, epoch, model, optimizer, best_psnr=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train final LMFA adaptive denoise + SE + edge loss model")
    parser.add_argument("--train_hazy", type=str, required=True)
    parser.add_argument("--train_clear", type=str, required=True)
    parser.add_argument("--val_hazy", type=str, required=True)
    parser.add_argument("--val_clear", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_final")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--alpha_ssim", type=float, default=0.02)
    parser.add_argument("--alpha_edge", type=float, default=0.1)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    train_set = RESIDEPairedDataset(
        hazy_dir=args.train_hazy,
        clear_dir=args.train_clear,
        crop_size=args.crop_size,
        training=True,
        add_noise_prob=0.4,
        noise_sigma_min=0.01,
        noise_sigma_max=0.05,
        patch_noise_prob=0.5,
    )

    val_set = RESIDEPairedDataset(
        hazy_dir=args.val_hazy,
        clear_dir=args.val_clear,
        crop_size=None,
        training=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = LMFANetAdaptiveDenoiseSE().to(device)
    criterion = CombinedLoss(alpha_ssim=args.alpha_ssim, alpha_edge=args.alpha_edge)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -1.0
    log_file = save_dir / "train_log.csv"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_gate,val_loss,val_psnr,val_ssim,val_gate,time_sec\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_gate = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr, val_ssim, val_gate = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Gate: {train_gate:.4f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.4f} | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Val Gate: {val_gate:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_gate:.4f},"
                f"{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{val_gate:.4f},{elapsed:.2f}\n"
            )

        latest_ckpt = save_dir / "latest.pth"
        save_checkpoint(latest_ckpt, epoch, model, optimizer, best_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ckpt = save_dir / "best.pth"
            save_checkpoint(best_ckpt, epoch, model, optimizer, best_psnr)
            print(f"Saved new best checkpoint: {best_ckpt}")

        if epoch % args.save_every == 0:
            epoch_ckpt = save_dir / f"epoch_{epoch}.pth"
            save_checkpoint(epoch_ckpt, epoch, model, optimizer, best_psnr)

    print(f"Training complete. Best PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()