"""
run2.py  –  Shoe detection via static background differencing.

Press 'b' to capture the background (make sure the pad is clear).
Everything that differs from that snapshot is treated as foreground.
Stationary shoes show up fine since they differ from the clean background.

Controls
────────
  b  – capture background (clear the pad first)
  s  – save region corners to config.yaml
  ESC – quit
"""

import cv2
import yaml
import os
import sys
import math
import numpy as np
from collections import deque
from pynput.keyboard import Controller, Key

CONFIG_FILE = "config.yaml"
WINDOW_NAME = "run2.py  –  b=capture bg  s=save  ESC=quit"

# Tunable constants
SMOOTH_WINDOW      = 5
MIN_BLOB_AREA      = 500    # px² — ignore tiny noise blobs
MAX_BLOB_AREA      = 80000  # px² — ignore huge blobs (whole body)
DIFF_THRESHOLD     = 30     # per-pixel diff threshold (0-255) — raise to reduce noise
MORPH_KERNEL_SIZE  = 9      # erosion/dilation to clean up mask

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
occlusion_map = config.get('occlusion_map', {})
occl_thresh   = config.get('occlusion_proximity_threshold', 0.15)

def get_key(key_str):
    special = {
        "space": Key.space, "enter": Key.enter,
        "up": Key.up, "down": Key.down,
        "left": Key.left, "right": Key.right,
    }
    return special.get(key_str.lower(), key_str)

key_to_idx = {reg['key']: i for i, reg in enumerate(regions_data)}

# ── Polygon helpers ────────────────────────────────────────────────────────────
def corners_to_px(corners, w, h):
    return np.array([[int(c[0] * w), int(c[1] * h)] for c in corners], np.int32)

def shrink_polygon(pts, margin):
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    scale  = 1.0 - margin * 2
    return np.array([
        [int(cx + (p[0] - cx) * scale), int(cy + (p[1] - cy) * scale)]
        for p in pts
    ], np.int32)

def point_in_polygon(pt, poly):
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0

# ── Key press helpers ──────────────────────────────────────────────────────────
key_is_down   = [False] * len(regions_data)
region_active = [False] * len(regions_data)

def press_region(idx):
    if not key_is_down[idx]:
        keyboard.press(get_key(regions_data[idx]['key']))
        key_is_down[idx] = True
        print(f"  [-] KEY DOWN: {regions_data[idx]['key'].upper()}")

def release_region(idx):
    if key_is_down[idx]:
        keyboard.release(get_key(regions_data[idx]['key']))
        key_is_down[idx] = False
        print(f"  [x] KEY UP:   {regions_data[idx]['key'].upper()}")

def release_all():
    for i in range(len(regions_data)):
        release_region(i)
    region_active[:] = [False] * len(regions_data)

# ── Background + colour calibration state ─────────────────────────────────────
background  = None   # captured clean background frame (grayscale)
leg_color   = None   # BGR sample clicked by user
shoe_color  = None   # BGR sample clicked by user
morph_kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

# setup_step: 'bg' → 'leg' → 'shoe' → None (running)
setup_step  = 'bg'
setup_frame = None   # frozen frame used during colour clicking

SAMPLE_RADIUS = 8    # px radius to average colour sample

# ── Smoothing buffers ──────────────────────────────────────────────────────────
shoe_history = [deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)]

# ── Camera + mouse ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cam_idx)
cv2.namedWindow(WINDOW_NAME)

dims = {
    'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
}

selected_region_idx = -1
selected_corner_idx = -1

def sample_color(frame, x, y, r=SAMPLE_RADIUS):
    """Mean BGR of a small patch around (x, y)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    patch = frame[y1:y2, x1:x2]
    return cv2.mean(patch)[:3]   # (B, G, R)

def mouse_handler(event, x, y, flags, param):
    global selected_region_idx, selected_corner_idx, leg_color, shoe_color, setup_step, setup_frame
    width, height = param['width'], param['height']

    if event == cv2.EVENT_LBUTTONDOWN:
        # ── Setup colour clicks ────────────────────────────────────────────────
        if setup_step == 'leg' and setup_frame is not None:
            leg_color = sample_color(setup_frame, x, y)
            setup_step = 'shoe'
            print(f"  ✔  Leg colour sampled: BGR={tuple(int(v) for v in leg_color)}")
            print("  Now click on your SHOE in the preview.")
            return
        if setup_step == 'shoe' and setup_frame is not None:
            shoe_color = sample_color(setup_frame, x, y)
            setup_step = None
            print(f"  ✔  Shoe colour sampled: BGR={tuple(int(v) for v in shoe_color)}")
            print("  ✅  Setup complete — running!\n")
            return

        # ── Normal corner dragging ─────────────────────────────────────────────
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

print("\n╔══════════════════════════════════════════════════════╗")
print("║  Setup:                                             ║")
print("║  1. Clear the pad, press 'b' to capture background ║")
print("║  2. Step on the pad, press 'c' to freeze frame     ║")
print("║  3. Click your LEG, then click your SHOE           ║")
print("║                                                      ║")
print("║  b   – capture background                           ║")
print("║  c   – freeze frame for leg/shoe colour click       ║")
print("║  s   – save region corners                          ║")
print("║  ESC – quit                                         ║")
print("╚══════════════════════════════════════════════════════╝\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
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

    # ── Background differencing ────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Step 1: capture background ─────────────────────────────────────────────
    if setup_step == 'bg' or background is None:
        cv2.putText(display, "Clear the pad then press 'b' to capture background",
                    (10, fh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k in (ord('b'), ord('B')):
            background = gray.copy()
            setup_step = 'colour'
            print("  ✅  Background captured.")
            print("  Now step onto the pad and press 'c' to freeze the frame.\n")
        continue

    # ── Step 2: freeze frame for colour clicking ───────────────────────────────
    if setup_step == 'colour':
        cv2.putText(display, "Stand on pad — press 'c' to freeze for colour picking",
                    (10, fh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k in (ord('c'), ord('C')):
            setup_frame = frame.copy()
            setup_step  = 'leg'
            print("  Frame frozen. Click your LEG in the preview window.")
        continue

    # ── Step 3: show frozen frame while user clicks leg then shoe ──────────────
    if setup_step in ('leg', 'shoe'):
        display = setup_frame.copy()
        if setup_step == 'leg':
            msg = "Click on your LEG"
        else:
            # Draw the leg sample marker
            msg = "Click on your SHOE"
        cv2.putText(display, msg, (10, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)
        continue

    diff    = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ── Pixel-level colour filter ──────────────────────────────────────────────
    # For each foreground pixel, keep it only if it's closer to shoe_color
    # than leg_color. This runs before morphology so legs can't merge with shoes.
    if shoe_color is not None and leg_color is not None:
        shoe_arr = np.array(shoe_color, dtype=np.float32)
        leg_arr  = np.array(leg_color,  dtype=np.float32)
        frame_f  = frame.astype(np.float32)                     # (H, W, 3)
        d_shoe   = np.linalg.norm(frame_f - shoe_arr, axis=2)  # (H, W)
        d_leg    = np.linalg.norm(frame_f - leg_arr,  axis=2)  # (H, W)
        shoe_px  = (d_shoe < d_leg).astype(np.uint8) * 255     # pixels closer to shoe
        mask     = cv2.bitwise_and(mask, shoe_px)

    # ------------------------------------------------------------------
    # Primary cleanup
    # ------------------------------------------------------------------

    fg_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kern)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kern)

    # ------------------------------------------------------------------
    # Optional blob splitting
    # Attempt to separate shoes connected by a thin bridge
    # ------------------------------------------------------------------

    SPLIT_KERNEL_SIZE = 16
    split_kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SPLIT_KERNEL_SIZE, SPLIT_KERNEL_SIZE)
    )

    # Original blob count
    orig_contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    orig_count = len(orig_contours)

    # Shrink blobs slightly to break thin connections
    split_mask = cv2.erode(
        fg_mask,
        split_kern,
        iterations=1
    )

    # Count blobs after erosion
    split_contours, _ = cv2.findContours(
        split_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    split_count = len(split_contours)
    split_max_area = max(
        (cv2.contourArea(c) for c in split_contours),
        default=0
    )

    # Only accept erosion if:
    # 1) it increased the number of blobs
    # 2) the resulting blobs are still large enough to be shoes
    if split_count > orig_count and split_max_area > MIN_BLOB_AREA:
        fg_mask = split_mask

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        blobs.append((area, cx, cy, cnt))

        # Keep the two largest blobs
        blobs.sort(key=lambda b: b[0], reverse=True)
        raw_centres = [(cx, cy) for _, cx, cy, _ in blobs[:2]]

        # ------------------------------------------------------------------
        # Tracking logic
        # ------------------------------------------------------------------

        if len(raw_centres) == 0:
            # Nothing visible -> clear tracking completely
            for buf in shoe_history:
                buf.clear()

        elif len(raw_centres) == 1:
            # Only one shoe visible.
            # Do NOT preserve the old second shoe position.
            # The occlusion map will infer the hidden foot.
            shoe_history[0].clear()
            shoe_history[1].clear()
            shoe_history[0].append(raw_centres[0])

        else:
            # Two blobs visible -> perform proximity matching
            last_known = [buf[-1] if buf else None for buf in shoe_history]

            assigned = [False] * len(raw_centres)

            for buf_idx, last in enumerate(last_known):
                if last is None:
                    continue

                best_dist = float('inf')
                best_det = -1

                for det_idx, pt in enumerate(raw_centres):
                    if assigned[det_idx]:
                        continue

                    d = math.hypot(pt[0] - last[0], pt[1] - last[1])

                    if d < best_dist:
                        best_dist = d
                        best_det = det_idx

                if best_det != -1:
                    shoe_history[buf_idx].append(raw_centres[best_det])
                    assigned[best_det] = True

            for det_idx, pt in enumerate(raw_centres):
                if not assigned[det_idx]:
                    for buf in shoe_history:
                        if not buf:
                            buf.append(pt)
                            break

    if len(raw_centres) < len([b for b in shoe_history if b]):
        if raw_centres:
            for buf in shoe_history:
                if not buf:
                    continue
                if not any(math.hypot(buf[-1][0] - pt[0], buf[-1][1] - pt[1]) < fw * 0.3
                           for pt in raw_centres):
                    buf.clear()
        else:
            for buf in shoe_history:
                buf.clear()

    # ── Outlier removal + averaging ────────────────────────────────────────────
    buf_pts = [np.array(buf, dtype=float) if buf else None for buf in shoe_history]
    shoe_centres = []
    for i, pts in enumerate(buf_pts):
        if pts is None:
            continue
        other_pts = next((p for j, p in enumerate(buf_pts) if j != i and p is not None), None)
        if other_pts is not None and len(pts) >= 3:
            other_mean = other_pts.mean(axis=0)
            dists_to_other = np.linalg.norm(pts - other_mean, axis=1)
            to_remove = np.argsort(dists_to_other)[:min(2, len(pts) - 1)]
            pts = np.delete(pts, to_remove, axis=0)
        sx, sy = pts.mean(axis=0)
        shoe_centres.append((int(sx), int(sy)))

    # ── Region detection ───────────────────────────────────────────────────────
    should_be_active = set()
    inferred_active  = set()
    too_close        = False

    if shoe_centres:
        detected_region_keys = set()

        for shoe_pt in shoe_centres:
            for idx, reg in enumerate(regions_data):
                outer_poly = corners_to_px(reg['corners'], fw, fh)
                inner_poly = shrink_polygon(outer_poly, enter_margin)
                if region_active[idx]:
                    if point_in_polygon(shoe_pt, outer_poly):
                        should_be_active.add(idx)
                        detected_region_keys.add(reg['key'])
                else:
                    if point_in_polygon(shoe_pt, inner_poly):
                        should_be_active.add(idx)
                        detected_region_keys.add(reg['key'])

        too_close = (
            len(shoe_centres) == 2 and
            math.hypot(
                shoe_centres[0][0] - shoe_centres[1][0],
                shoe_centres[0][1] - shoe_centres[1][1],
            ) < fw * occl_thresh
        )
        if len(shoe_centres) == 1:
            for visible_key in list(detected_region_keys):
                inferred_key = occlusion_map.get(str(visible_key))
                if inferred_key and inferred_key in key_to_idx:
                    inferred_idx = key_to_idx[inferred_key]
                    should_be_active.add(inferred_idx)
                    inferred_active.add(inferred_idx)

    for idx in range(len(regions_data)):
        if idx in should_be_active:
            region_active[idx] = True
            press_region(idx)
        else:
            region_active[idx] = False
            release_region(idx)

    # ── Draw blobs ─────────────────────────────────────────────────────────────
    for _, cx, cy, cnt in blobs:
        cv2.drawContours(display, [cnt], -1, (0, 255, 255), 2)
        cv2.putText(display, "shoe", (cx - 12, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ── Draw overlay ───────────────────────────────────────────────────────────
    for idx, reg in enumerate(regions_data):
        outer_poly = corners_to_px(reg['corners'], fw, fh)
        inner_poly = shrink_polygon(outer_poly, enter_margin)

        if idx in inferred_active:
            color = (0, 100, 255)
            label = f"{reg['key'].upper()} ▶ INFERRED"
        elif region_active[idx]:
            color = (0, 0, 255)
            label = f"{reg['key'].upper()} ▶ ACTIVE"
        else:
            color = (0, 255, 0)
            label = reg['key'].upper()

        cv2.polylines(display, [outer_poly.reshape(-1, 1, 2)], True, color, 2)
        cv2.polylines(display, [inner_poly.reshape(-1, 1, 2)], True, color, 1)
        tx = int(reg['corners'][0][0] * fw)
        ty = int(reg['corners'][0][1] * fh) - 10
        cv2.putText(display, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw shoe centres
    for sx, sy in shoe_centres:
        cv2.drawMarker(display, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(display, (sx, sy), 6, (0, 255, 255), -1)

    if too_close and len(shoe_centres) == 2:
        cv2.line(display, shoe_centres[0], shoe_centres[1], (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "OCCLUDED",
                    ((shoe_centres[0][0] + shoe_centres[1][0]) // 2 - 40,
                     (shoe_centres[0][1] + shoe_centres[1][1]) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)

    # Draw fg mask inset — shows pixel-level colour filtered result
    mask_small = cv2.resize(fg_mask, (fw // 4, fh // 4))
    mask_bgr   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    display[fh - fh//4:, fw - fw//4:] = mask_bgr

    cv2.putText(display, f"Shoes: {len(shoe_centres)}", (10, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord('b'), ord('B')):
        background = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        setup_step = 'colour'
        for buf in shoe_history:
            buf.clear()
        print("  ✅  Background recaptured. Step on pad and press 'c' to re-pick colours.")
    elif k in (ord('c'), ord('C')):
        setup_frame = frame.copy()
        setup_step  = 'leg'
        for buf in shoe_history:
            buf.clear()
        print("  Frame frozen. Click your LEG in the preview window.")
    elif k in (ord('s'), ord('S')):
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("  💾  Config saved →", CONFIG_FILE)

# ── Cleanup ────────────────────────────────────────────────────────────────────
release_all()
cap.release()
cv2.destroyAllWindows()