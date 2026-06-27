"""
run.py  –  Load trained shoe model, detect shoe centres, trigger regions.

Detection logic
───────────────
  • Run ShoeNet on each frame → heatmap → find up to 2 peaks (shoe centres).
  • A region is ENTERED  when a shoe centre is inside the inner polygon  (shrunk by enter_margin).
  • A region is EXITED   when the shoe centre leaves the outer polygon   (the original corners).
  • This hysteresis prevents flickering at borders.
  • If only 1 shoe is detected, the occluded shoe's region is inferred from
    config['occlusion_map'] keyed by the visible shoe's region key.
  • If 0 shoes are detected, all keys are released.

Config additions (optional, add to config.yaml)
────────────────────────────────────────────────
  enter_margin: 0.08        # fraction to shrink region for enter trigger (default 0.08)
  occlusion_map:            # when only 1 shoe visible, also activate these
    '1': '3'                # if shoe in region 1, also press region 3
    '3': '1'
"""

import cv2
import yaml
import os
import sys
import math
import numpy as np
import time
from collections import deque
from pynput.keyboard import Controller, Key

# ── Optional torch ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    print("ERROR: PyTorch not found.  Run:  pip install torch torchvision pillow")
    sys.exit(1)

CONFIG_FILE = "config.yaml"
MODEL_FILE  = "shoe_model.pt"
WINDOW_NAME = "run.py  –  ESC to quit"
INPUT_W = INPUT_H = 224

keyboard = Controller()

# ── Load config ────────────────────────────────────────────────────────────────
if not os.path.exists(CONFIG_FILE):
    print(f"ERROR: {CONFIG_FILE} not found.")
    sys.exit(1)
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

cam_idx       = config['camera']['device_index']
mirror        = config['camera']['mirror_preview']
regions_data  = config['regions']
enter_margin  = config.get('enter_margin', 0.08)
occlusion_map = config.get('occlusion_map', {})   # key → key

def get_key(key_str):
    special = {
        "space": Key.space, "enter": Key.enter,
        "up": Key.up, "down": Key.down,
        "left": Key.left, "right": Key.right,
    }
    return special.get(key_str.lower(), key_str)

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: {MODEL_FILE} not found.  Run train.py first.")
    sys.exit(1)

class ShoeNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        self.enc1 = block(3,   16, stride=2)
        self.enc2 = block(16,  32, stride=2)
        self.enc3 = block(32,  64, stride=2)
        self.enc4 = block(64, 128, stride=2)
        self.up4  = nn.Sequential(nn.Upsample(scale_factor=2), block(128, 64))
        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2), block(64,  32))
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2), block(32,  16))
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2), block(16,   8))
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        x = self.up1(self.up2(self.up3(self.up4(x))))
        return torch.sigmoid(self.head(x))

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt      = torch.load(MODEL_FILE, map_location=device)
model     = ShoeNet().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  ✅  Model loaded from {MODEL_FILE}  (device: {device})")

img_transform = T.Compose([
    T.Resize((INPUT_H, INPUT_W)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Polygon helpers ────────────────────────────────────────────────────────────
def corners_to_px(corners, w, h):
    return np.array([[int(c[0] * w), int(c[1] * h)] for c in corners], np.int32)

def shrink_polygon(pts, margin):
    """Return a polygon shrunk inward by `margin` fraction of its bounding box."""
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    bw = pts[:, 0].max() - pts[:, 0].min()
    bh = pts[:, 1].max() - pts[:, 1].min()
    scale = 1.0 - margin * 2
    shrunk = np.array([
        [cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale]
        for p in pts
    ], np.int32)
    return shrunk

def point_in_polygon(pt, poly):
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0

# ── Heatmap → shoe centres ─────────────────────────────────────────────────────
def find_peaks(heatmap_np, frame_w, frame_h, threshold=0.3, min_dist=40):
    """
    Find up to 2 local maxima in the heatmap and map them back to frame coords.
    Returns list of (frame_x, frame_y) tuples, 0–2 items.
    """
    hm = heatmap_np.copy()
    peaks = []
    for _ in range(2):
        idx   = np.unravel_index(np.argmax(hm), hm.shape)
        val   = hm[idx]
        if val < threshold:
            break
        py, px = idx
        # Map from heatmap space → frame space
        fx = int(px / INPUT_W * frame_w)
        fy = int(py / INPUT_H * frame_h)
        peaks.append((fx, fy))
        # Suppress this peak so we find the next one
        r = min_dist // 2
        y1 = max(0, py - r); y2 = min(hm.shape[0], py + r)
        x1 = max(0, px - r); x2 = min(hm.shape[1], px + r)
        hm[y1:y2, x1:x2] = 0.0
    return peaks

# ── Runtime state ──────────────────────────────────────────────────────────────
key_is_down    = [False] * len(regions_data)
region_active  = [False] * len(regions_data)   # hysteresis state

selected_region_idx = -1
selected_corner_idx = -1

SMOOTH_WINDOW = 5
shoe_history  = [deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)]  # one per shoe slot

def press_region(idx):
    if not key_is_down[idx]:
        k = get_key(regions_data[idx]['key'])
        keyboard.press(k)
        key_is_down[idx] = True
        print(f"  [-] KEY DOWN: {regions_data[idx]['key'].upper()}")

def release_region(idx):
    if key_is_down[idx]:
        k = get_key(regions_data[idx]['key'])
        keyboard.release(k)
        key_is_down[idx] = False
        print(f"  [x] KEY UP:   {regions_data[idx]['key'].upper()}")

def release_all():
    for i in range(len(regions_data)):
        release_region(i)
    region_active[:] = [False] * len(regions_data)

# ── Occlusion map: key string → region index ───────────────────────────────────
key_to_idx = {reg['key']: i for i, reg in enumerate(regions_data)}

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cam_idx)
cv2.namedWindow(WINDOW_NAME)

dims = {
    'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
}

def mouse_handler(event, x, y, flags, param):
    global selected_region_idx, selected_corner_idx
    width, height = param['width'], param['height']
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = 15
        for r_idx, reg in enumerate(regions_data):
            for c_idx, corner in enumerate(reg['corners']):
                cx, cy = int(corner[0] * width), int(corner[1] * height)
                dist = math.hypot(x - cx, y - cy)
                if dist < min_dist:
                    selected_region_idx = r_idx
                    selected_corner_idx = c_idx
                    min_dist = dist
    elif event == cv2.EVENT_MOUSEMOVE and selected_region_idx != -1:
        regions_data[selected_region_idx]['corners'][selected_corner_idx] = [
            max(0.0, min(1.0, x / width)),
            max(0.0, min(1.0, y / height)),
        ]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_region_idx = -1
        selected_corner_idx = -1

cv2.setMouseCallback(WINDOW_NAME, mouse_handler, dims)
print("  Running — drag corners to adjust regions, s=save, ESC=quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    if mirror:
        frame = cv2.flip(frame, 1)

    fh, fw = frame.shape[:2]
    dims['width'], dims['height'] = fw, fh
    display = frame.copy()

    # ── Run model ──────────────────────────────────────────────────────────────
    pil_img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp      = img_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmap = model(inp)[0, 0].cpu().numpy()   # (INPUT_H, INPUT_W)

    shoe_centres = find_peaks(heatmap, fw, fh)
    too_close = False

    # ── Smooth positions — proximity matched to avoid slot swapping ───────────
    # Get last known position for each buffer (or None if empty)
    last_known = [buf[-1] if buf else None for buf in shoe_history]

    # Match each new detection to the nearest buffer by last known position
    assigned = [False] * len(shoe_centres)
    for buf_idx, last in enumerate(last_known):
        if last is None:
            continue
        best_dist, best_det = float('inf'), -1
        for det_idx, pt in enumerate(shoe_centres):
            if assigned[det_idx]:
                continue
            d = math.hypot(pt[0] - last[0], pt[1] - last[1])
            if d < best_dist:
                best_dist, best_det = d, det_idx
        if best_det != -1:
            shoe_history[buf_idx].append(shoe_centres[best_det])
            assigned[best_det] = True

    # Any unmatched detections go to empty buffers
    for det_idx, pt in enumerate(shoe_centres):
        if not assigned[det_idx]:
            for buf in shoe_history:
                if not buf:
                    buf.append(pt)
                    break

    # Clear buffers for shoes that disappeared
    if len(shoe_centres) < len([b for b in shoe_history if b]):
        # Find the buffer whose last point is furthest from any detection — evict it
        if shoe_centres:
            for buf in shoe_history:
                if not buf:
                    continue
                if not any(
                    math.hypot(buf[-1][0] - pt[0], buf[-1][1] - pt[1]) < fw * 0.3
                    for pt in shoe_centres
                ):
                    buf.clear()
        else:
            for buf in shoe_history:
                buf.clear()

    smoothed_centres = []
    buf_pts = [np.array(buf, dtype=float) if buf else None for buf in shoe_history]

    for i, pts in enumerate(buf_pts):
        if pts is None:
            continue
        # Find the other shoe's current mean (if it exists)
        other_pts = next((p for j, p in enumerate(buf_pts) if j != i and p is not None), None)
        if other_pts is not None and len(pts) >= 3:
            other_mean = other_pts.mean(axis=0)
            dists_to_other = np.linalg.norm(pts - other_mean, axis=1)
            to_remove = np.argsort(dists_to_other)[:min(2, len(pts) - 1)]
            pts = np.delete(pts, to_remove, axis=0)
        sx, sy = pts.mean(axis=0)
        smoothed_centres.append((int(sx), int(sy)))
    shoe_centres = smoothed_centres

    # ── Build set of regions that should be active this frame ─────────────────
    should_be_active = set()
    inferred_active  = set()

    if len(shoe_centres) == 0:
        # No shoes detected — release everything
        pass

    else:
        # Determine which regions each detected shoe centre falls into
        detected_region_keys = set()

        for shoe_pt in shoe_centres:
            for idx, reg in enumerate(regions_data):
                outer_poly = corners_to_px(reg['corners'], fw, fh)
                inner_poly = shrink_polygon(outer_poly, enter_margin)

                if region_active[idx]:
                    # Already active: stay active until shoe leaves outer polygon
                    if point_in_polygon(shoe_pt, outer_poly):
                        should_be_active.add(idx)
                        detected_region_keys.add(reg['key'])
                else:
                    # Not active: only activate when shoe enters inner polygon
                    if point_in_polygon(shoe_pt, inner_poly):
                        should_be_active.add(idx)
                        detected_region_keys.add(reg['key'])

        # ── Occlusion inference ────────────────────────────────────────────────
        too_close = (
            len(shoe_centres) == 2 and
            math.hypot(
                shoe_centres[0][0] - shoe_centres[1][0],
                shoe_centres[0][1] - shoe_centres[1][1],
            ) < fw * config.get('occlusion_proximity_threshold', 0.15)
        )
        if len(shoe_centres) == 1 or too_close:
            for visible_key in list(detected_region_keys):
                inferred_key = occlusion_map.get(str(visible_key))
                if inferred_key and inferred_key in key_to_idx:
                    inferred_idx = key_to_idx[inferred_key]
                    should_be_active.add(inferred_idx)
                    inferred_active.add(inferred_idx)

    # ── Apply state changes ────────────────────────────────────────────────────
    for idx in range(len(regions_data)):
        if idx in should_be_active:
            region_active[idx] = True
            press_region(idx)
        else:
            region_active[idx] = False
            release_region(idx)

    # ── Draw overlay ───────────────────────────────────────────────────────────
    for idx, reg in enumerate(regions_data):
        outer_poly = corners_to_px(reg['corners'], fw, fh)
        inner_poly = shrink_polygon(outer_poly, enter_margin)

        if idx in inferred_active:
            color = (0, 100, 255)   # orange — inferred via occlusion
            label = f"{reg['key'].upper()} ▶ INFERRED"
        elif region_active[idx]:
            color = (0, 0, 255)     # red — directly triggered
            label = f"{reg['key'].upper()} ▶ ACTIVE"
        else:
            color = (0, 255, 0)     # green — idle
            label = reg['key'].upper()

        cv2.polylines(display, [outer_poly.reshape(-1, 1, 2)], True, color, 2)
        cv2.polylines(display, [inner_poly.reshape(-1, 1, 2)], True, color, 1)

        tx = int(reg['corners'][0][0] * fw)
        ty = int(reg['corners'][0][1] * fh) - 10
        cv2.putText(display, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw shoe centre markers
    for sx, sy in shoe_centres:
        cv2.drawMarker(display, (sx, sy), (0, 255, 255),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(display, (sx, sy), 6, (0, 255, 255), -1)

    if too_close and len(shoe_centres) == 2:
        cv2.line(display, shoe_centres[0], shoe_centres[1], (0, 100, 255), 2, cv2.LINE_AA)
        # cv2.putText(display, "OCCLUDED", (
        #                 (shoe_centres[0][0] + shoe_centres[1][0]) // 2 - 40,
        #                 (shoe_centres[0][1] + shoe_centres[1][1]) // 2 - 10,
        #             ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)

    shoe_count_text = f"Shoes detected: {len(shoe_centres)}"
    cv2.putText(display, shoe_count_text, (10, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord('s'), ord('S')):
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("  💾  Config saved →", CONFIG_FILE)

# ── Cleanup ────────────────────────────────────────────────────────────────────
release_all()
cap.release()
cv2.destroyAllWindows()