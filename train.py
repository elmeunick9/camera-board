"""
train.py  –  Capture frames, tag shoe centres with clicks, train a tiny CNN.

Workflow
────────
1.  Press SPACE to capture a frame (aim for ~10 varied frames).
2.  For each captured frame, click the centre of each visible shoe (0, 1 or 2 clicks).
3.  Press ENTER when done tagging a frame.  Press ESC to skip a frame.
4.  When all frames are tagged, press 't' to train and save the model.

Output
──────
  data/frames/   – captured PNG frames
  data/labels.json – {filename: [[x,y], ...]} normalised 0-1 coords
  shoe_model.pt  – trained model (loaded by run.py)
"""

import cv2
import json
import os
import time
import yaml
import numpy as np

# ── Optional torch import with friendly error ──────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from torchvision.transforms import functional as TF
    from PIL import Image
except ImportError:
    print("ERROR: PyTorch not found.  Run:  pip install torch torchvision pillow")
    exit(1)

CONFIG_FILE  = "config.yaml"
FRAMES_DIR   = "data/frames"
LABELS_FILE  = "data/labels.json"
MODEL_FILE   = "shoe_model.pt"
WINDOW_NAME  = "train.py  –  SPACE=capture  ENTER=done tagging  t=train  ESC=quit"

INPUT_W, INPUT_H = 224, 224   # model input resolution
EPOCHS           = 50
BATCH_SIZE       = 8
LR               = 1e-3
HEATMAP_SIGMA    = 15         # gaussian spread for target heatmaps (pixels at INPUT res)

os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Load config ────────────────────────────────────────────────────────────────
if not os.path.exists(CONFIG_FILE):
    print(f"ERROR: {CONFIG_FILE} not found.")
    exit(1)
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

cam_idx = config['camera']['device_index']
mirror  = config['camera']['mirror_preview']

# ── Load existing labels ───────────────────────────────────────────────────────
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE) as f:
        labels: dict = json.load(f)
    print(f"  Loaded {len(labels)} existing labels from {LABELS_FILE}")
else:
    labels = {}

# ── State ──────────────────────────────────────────────────────────────────────
current_frame      = None
current_frame_name = None
pending_clicks     = []   # list of (norm_x, norm_y) for current frame
mode               = 'capture'   # 'capture' | 'tag'

def save_labels():
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"  💾  Labels saved → {LABELS_FILE}  ({len(labels)} frames)")

# ── Mouse handler ──────────────────────────────────────────────────────────────
def mouse_handler(event, x, y, flags, param):
    if mode != 'tag':
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = param['shape']
        nx, ny = x / w, y / h
        pending_clicks.append([nx, ny])
        print(f"  📍  Click {len(pending_clicks)}: ({nx:.3f}, {ny:.3f})")

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cam_idx)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
frame_shape = {'shape': (480, 640)}
cv2.setMouseCallback(WINDOW_NAME, mouse_handler, frame_shape)

print("\n╔══════════════════════════════════════════════════╗")
print("║  SPACE  – capture frame                         ║")
print("║  click  – mark a shoe centre (up to 2)          ║")
print("║  ENTER  – confirm tags and move to next frame   ║")
print("║  ESC    – skip / discard current frame          ║")
print("║  t      – train model on all tagged frames      ║")
print("║  q      – quit without training                 ║")
print("╚══════════════════════════════════════════════════╝\n")
print(f"  {len(labels)} frames already tagged.  Capture more or press 't' to train.\n")

while True:
    ret, live = cap.read()
    if not ret:
        break
    if mirror:
        live = cv2.flip(live, 1)

    h, w = live.shape[:2]
    frame_shape['shape'] = (h, w)
    display = live.copy()

    if mode == 'capture':
        cv2.putText(display, f"CAPTURE MODE  |  {len(labels)} frames tagged  |  SPACE to capture",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    elif mode == 'tag':
        # Draw frozen frame with click markers
        display = current_frame.copy()
        for i, (nx, ny) in enumerate(pending_clicks):
            px, py = int(nx * w), int(ny * h)
            cv2.drawMarker(display, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(display, f"shoe {i+1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display,
                    f"TAG  |  {len(pending_clicks)} shoe(s) marked  |  ENTER=confirm  ESC=skip",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 27:   # ESC
        if mode == 'tag':
            print(f"  ✘  Skipped {current_frame_name}")
            mode = 'capture'
            pending_clicks.clear()

    elif key == ord(' ') and mode == 'capture':
        ts   = time.strftime("%Y%m%d_%H%M%S")
        fname = f"frame_{ts}.png"
        fpath = os.path.join(FRAMES_DIR, fname)
        cv2.imwrite(fpath, live)
        current_frame      = live.copy()
        current_frame_name = fname
        pending_clicks.clear()
        mode = 'tag'
        print(f"  📸  Captured → {fpath}  (click shoe centres, then ENTER)")

    elif key == 13 and mode == 'tag':   # ENTER
        labels[current_frame_name] = pending_clicks.copy()
        save_labels()
        print(f"  ✔  {current_frame_name}: {len(pending_clicks)} shoe(s) tagged")
        mode = 'capture'
        pending_clicks.clear()

    elif key == ord('t'):
        # ── TRAINING ──────────────────────────────────────────────────────────
        if len(labels) < 3:
            print("  ✘  Need at least 3 tagged frames to train.  Capture more.")
            continue

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n━━━  TRAINING  ━━━  ({len(labels)} frames)\n")

        # ── Gaussian heatmap helper ────────────────────────────────────────────
        def make_heatmap(points_norm, H=INPUT_H, W=INPUT_W, sigma=HEATMAP_SIGMA):
            """Single-channel heatmap with one gaussian per shoe point."""
            hm = np.zeros((H, W), dtype=np.float32)
            for nx, ny in points_norm:
                cx, cy = int(nx * W), int(ny * H)
                for y in range(H):
                    for x in range(W):
                        hm[y, x] += np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            return np.clip(hm, 0, 1)

        # ── Vectorised gaussian (much faster) ─────────────────────────────────
        ys = np.arange(INPUT_H, dtype=np.float32)
        xs = np.arange(INPUT_W, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        def make_heatmap_fast(points_norm):
            hm = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
            for nx, ny in points_norm:
                cx, cy = nx * INPUT_W, ny * INPUT_H
                hm += np.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * HEATMAP_SIGMA**2))
            return np.clip(hm, 0, 1)

        # ── Dataset ────────────────────────────────────────────────────────────
        class ShoeDataset(Dataset):
            def __init__(self, labels_dict, frames_dir):
                self.items = list(labels_dict.items())
                self.frames_dir = frames_dir
                self.img_tf = T.Compose([
                    T.Resize((INPUT_H, INPUT_W)),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                fname, pts = self.items[i]
                img_path = os.path.join(self.frames_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img_t = self.img_tf(img)
                hm = make_heatmap_fast(pts) if pts else np.zeros((INPUT_H, INPUT_W), np.float32)
                hm_t = torch.from_numpy(hm).unsqueeze(0)   # (1, H, W)
                return img_t, hm_t

        dataset    = ShoeDataset(labels, FRAMES_DIR)
        dataloader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)),
                                shuffle=True, drop_last=False)

        # ── Model  (tiny encoder-decoder) ──────────────────────────────────────
        class ShoeNet(nn.Module):
            def __init__(self):
                super().__init__()
                def block(cin, cout, stride=1):
                    return nn.Sequential(
                        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(cout),
                        nn.ReLU(inplace=True),
                    )
                # Encoder
                self.enc1 = block(3,   16, stride=2)   # 112
                self.enc2 = block(16,  32, stride=2)   # 56
                self.enc3 = block(32,  64, stride=2)   # 28
                self.enc4 = block(64, 128, stride=2)   # 14
                # Decoder
                self.up4  = nn.Sequential(nn.Upsample(scale_factor=2), block(128, 64))   # 28
                self.up3  = nn.Sequential(nn.Upsample(scale_factor=2), block(64,  32))   # 56
                self.up2  = nn.Sequential(nn.Upsample(scale_factor=2), block(32,  16))   # 112
                self.up1  = nn.Sequential(nn.Upsample(scale_factor=2), block(16,   8))   # 224
                self.head = nn.Conv2d(8, 1, 1)

            def forward(self, x):
                x = self.enc4(self.enc3(self.enc2(self.enc1(x))))
                x = self.up1(self.up2(self.up3(self.up4(x))))
                return torch.sigmoid(self.head(x))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")

        model     = ShoeNet().to(device)
        optimiser = optim.Adam(model.parameters(), lr=LR)
        loss_fn   = nn.BCELoss()

        model.train()
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            for imgs, hms in dataloader:
                imgs, hms = imgs.to(device), hms.to(device)
                optimiser.zero_grad()
                pred = model(imgs)
                loss = loss_fn(pred, hms)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(dataloader):.4f}")

        torch.save({
            'model_state': model.state_dict(),
            'input_size': (INPUT_W, INPUT_H),
            'heatmap_sigma': HEATMAP_SIGMA,
        }, MODEL_FILE)
        print(f"\n  ✅  Model saved → {MODEL_FILE}\n")
        break

cap.release()
cv2.destroyAllWindows()
