import cv2
import torch
import torchvision.transforms as T
from pathlib import Path
import sys

# IMPORTANT: add project path
sys.path.append(str(Path(__file__).resolve().parent))

from models.lmfanet import LMFANet


# ---------------- SETTINGS ----------------
VIDEO_PATH = "assets/fog_video3.mp4"
MODEL_PATH = "checkpoints/baseline/best.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = LMFANet().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)

if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()

to_tensor = T.ToTensor()

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (VERY IMPORTANT)
    frame = cv2.resize(frame, (640, 360))

    # Convert to tensor
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = to_tensor(img_rgb).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(inp)

    # Convert back
    pred = pred.squeeze(0).cpu().clamp(0, 1)
    pred = (pred.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    # Match LMFA cropping
    h, w = frame.shape[:2]
    orig_crop = frame[5:h-5, 5:w-5]

    # Resize output to match cropped input
    pred = cv2.resize(pred, (orig_crop.shape[1], orig_crop.shape[0]))

    # Side-by-side
    combined = cv2.hconcat([orig_crop, pred])

    cv2.imshow("Original vs Baseline LMFA", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()