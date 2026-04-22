import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lmfanet import LMFANet
from utils.dataset import RESIDEPairedDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_psnr_ssim


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for hazy, clear, _ in pbar:
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(hazy)

        if pred.shape != clear.shape:
            raise RuntimeError(
                f"Shape mismatch during training: pred={pred.shape}, clear={clear.shape}"
            )

        loss = criterion(pred, clear)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for hazy, clear, _ in tqdm(loader, desc="Validation", leave=False):
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        pred = model(hazy)

        if pred.shape != clear.shape:
            raise RuntimeError(
                f"Shape mismatch during validation: pred={pred.shape}, clear={clear.shape}"
            )

        for b in range(pred.size(0)):
            psnr, ssim = compute_psnr_ssim(pred[b], clear[b])
            total_psnr += psnr
            total_ssim += ssim
            count += 1

    return total_psnr / count, total_ssim / count


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
    parser = argparse.ArgumentParser(description="Train LMFA-Net")
    parser.add_argument("--train_hazy", type=str, required=True)
    parser.add_argument("--train_clear", type=str, required=True)
    parser.add_argument("--val_hazy", type=str, required=True)
    parser.add_argument("--val_clear", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=5)
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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = LMFANet().to(device)
    criterion = CombinedLoss(alpha_ssim=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -1.0
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,val_psnr,val_ssim,seconds\n", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        seconds = round(time.time() - start, 2)

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val PSNR: {val_psnr:.4f} | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Time: {seconds}s"
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{seconds}\n")

        save_checkpoint(save_dir / "latest.pth", epoch, model, optimizer, best_psnr)

        if epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch}.pth", epoch, model, optimizer, best_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(save_dir / "best.pth", epoch, model, optimizer, best_psnr)
            print(f"Saved new best checkpoint: {save_dir / 'best.pth'}")

    print("Training complete.")
    print(f"Best PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()
