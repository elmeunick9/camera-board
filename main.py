import cv2
import numpy as np
import yaml
import os
import math
import time
from collections import deque
from pynput.keyboard import Controller, Key

keyboard = Controller()
WINDOW_NAME = "DIY Motion Pad - Final 100ms Edition"
CONFIG_FILE = "config.yaml"

# ── Load Config ────────────────────────────────────────────────────────────────
if not os.path.exists(CONFIG_FILE):
    print(f"Error: {CONFIG_FILE} not found.")
    exit()
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

cam_idx        = config['camera']['device_index']
mirror         = config['camera']['mirror_preview']
active_color   = tuple(config['settings']['active_box_color'])
regions_data   = config['regions']

# ── Runtime State ──────────────────────────────────────────────────────────────
kernel_history      = {}
visual_flash_timers = [0]   * len(regions_data)
last_trigger_times  = [0.0] * len(regions_data)
foot_present        = [False] * len(regions_data)
key_is_down         = [False] * len(regions_data)

selected_region_idx = -1
selected_corner_idx = -1

# Calibration state machine
#   'idle'        – normal operation
#   'bg_pending'  – 'b' was pressed, waiting for the live frame to be captured
#   'fg_waiting'  – 'f' was pressed, waiting for the user to step and press a key
fg_calibration_mode = False   # kept for overlay colouring
calibration_state   = 'idle'

last_frame_time = time.time()

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_key(key_str):
    special_keys = {
        "space": Key.space, "enter": Key.enter,
        "up": Key.up, "down": Key.down,
        "left": Key.left, "right": Key.right,
    }
    return special_keys.get(key_str.lower(), key_str)

def parse_opencv_key(key_code):
    if key_code in (81, 2424832): return "left"
    if key_code in (82, 2490368): return "up"
    if key_code in (83, 2555904): return "right"
    if key_code in (84, 2621440): return "down"
    try:    return chr(key_code & 0xFF).lower()
    except ValueError: return ""

def sample_region_hsv(hsv_frame, reg, width, height):
    """Return the mean HSV of the polygon region, or None if mask is empty."""
    pts = np.array(
        [[int(p[0] * width), int(p[1] * height)] for p in reg['corners']],
        np.int32,
    )
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) == 0:
        return None
    mean = cv2.mean(hsv_frame, mask=mask)
    return [int(x) for x in mean[:3]]

def calibration_status_summary():
    """Print a tidy table of which regions are calibrated."""
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │           CALIBRATION STATUS SUMMARY            │")
    print("  ├──────────┬──────────────┬───────────────────────┤")
    print("  │  Region  │  Background  │      Foreground       │")
    print("  ├──────────┼──────────────┼───────────────────────┤")
    for reg in regions_data:
        key   = reg['key'].upper().center(8)
        bg_ok = "  ✔  OK  " if reg.get('background_hsv') else "  ✘ MISS "
        fg_ok = "  ✔  OK         " if reg.get('foreground_hsv') else "  ✘ NOT SET      "
        print(f"  │ {key} │{bg_ok}     │{fg_ok}│")
    print("  └──────────┴──────────────┴───────────────────────┘\n")

def print_controls():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         ⚡  100ms PRO TUNED SYSTEM  ⚡               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  b         – calibrate BACKGROUND (clear the pad)   ║")
    print("║  f         – calibrate FOREGROUND  (step on a pad)  ║")
    print("║  s         – save config to disk                    ║")
    print("║  ESC       – quit                                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

# ── Camera + Window ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cam_idx)
cv2.namedWindow(WINDOW_NAME)
dims = {
    'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
}

def mouse_handler(event, x, y, flags, param):
    global selected_region_idx, selected_corner_idx, regions_data
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
print_controls()
calibration_status_summary()

# ── Main Loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    current_time = time.time()
    dt = max(current_time - last_frame_time, 0.001)
    last_frame_time = current_time

    if mirror:
        frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    dims['width'], dims['height'] = width, height

    # HSV conversion happens once per frame — used by both detection and calibration
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Per-region detection ───────────────────────────────────────────────────
    for idx, reg in enumerate(regions_data):
        pts = np.array(
            [[int(p[0] * width), int(p[1] * height)] for p in reg['corners']],
            np.int32,
        ).reshape((-1, 1, 2))
        rx, ry, rw, rh = cv2.boundingRect(pts)

        target_key  = get_key(reg['key'])
        thresh_limit = reg.get('affinity_threshold', 50)

        max_box_presence   = 0.0
        active_motion_detected = False
        max_smooth_velocity    = -9999.0
        kernel_size = 16
        stride      = 16

        MOTION_VELOCITY_THRESH = 100
        TAP_DURATION_LIMIT     = 0.200
        HISTORY_WINDOW_SEC     = 0.100

        color_draw  = (0, 165, 255)   # Orange = uncalibrated
        status_text = "UNCALIBRATED — press 'b' then 'f'"

        if reg.get('background_hsv') and reg.get('foreground_hsv'):
            bg = np.array(reg['background_hsv'])
            fg = np.array(reg['foreground_hsv'])

            for ky in range(ry, ry + rh - kernel_size, stride):
                for kx in range(rx, rx + rw - kernel_size, stride):
                    cx, cy = kx + kernel_size // 2, ky + kernel_size // 2
                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        kernel_roi = hsv_frame[ky:ky+kernel_size, kx:kx+kernel_size]
                        curr = np.array(cv2.mean(kernel_roi)[:3])

                        d_bg = np.linalg.norm(curr - bg)
                        d_fg = np.linalg.norm(curr - fg)
                        tot  = d_bg + d_fg

                        if tot > 0:
                            k_affinity = (d_bg / tot) * 100
                            max_box_presence = max(max_box_presence, k_affinity)

                            mem_key = (idx, kx, ky)
                            if mem_key not in kernel_history:
                                kernel_history[mem_key] = deque(maxlen=10)
                            kernel_history[mem_key].append((current_time, k_affinity))

                            while (len(kernel_history[mem_key]) > 1 and
                                   current_time - kernel_history[mem_key][0][0] > HISTORY_WINDOW_SEC):
                                kernel_history[mem_key].popleft()

                            if len(kernel_history[mem_key]) > 1:
                                oldest_time, oldest_aff = kernel_history[mem_key][0]
                                time_span = max(current_time - oldest_time, 0.001)
                                smooth_k_velocity = abs(k_affinity - oldest_aff) / time_span
                            else:
                                smooth_k_velocity = 0.0

                            max_smooth_velocity = max(max_smooth_velocity, smooth_k_velocity)

                            if k_affinity >= thresh_limit and smooth_k_velocity > MOTION_VELOCITY_THRESH:
                                active_motion_detected = True

            # ── State machine ──────────────────────────────────────────────────
            color_draw  = (0, 255, 0)
            status_text = "IDLE"

            is_foot_here_now = (max_box_presence >= thresh_limit)

            if is_foot_here_now and not foot_present[idx]:
                last_trigger_times[idx] = current_time

            foot_present[idx] = is_foot_here_now
            time_held = current_time - last_trigger_times[idx]

            if foot_present[idx]:
                if not key_is_down[idx] and time_held >= TAP_DURATION_LIMIT:
                    color_draw  = (0, 100, 0)
                    status_text = "RESTING"

                elif not key_is_down[idx]:
                    keyboard.press(target_key)
                    key_is_down[idx] = True
                    print(f"  [-] KEY DOWN (Fresh Step): {reg['key'].upper()}")
                    color_draw  = active_color
                    status_text = "TAP TRIGGER"

                elif active_motion_detected:
                    last_trigger_times[idx] = current_time
                    color_draw  = (255, 255, 0)
                    status_text = "HOLD SUSTAINED"
                    if not key_is_down[idx]:
                        keyboard.press(target_key)
                        key_is_down[idx] = True
                        print(f"  [-] KEY DOWN (Hold Resumed): {reg['key'].upper()}")

                else:
                    if time_held >= TAP_DURATION_LIMIT and config.get('auto_release', True):
                        if key_is_down[idx]:
                            keyboard.release(target_key)
                            key_is_down[idx] = False
                            print(f"  [x] KEY UP (Auto-Released): {reg['key'].upper()}")
                        color_draw  = (0, 100, 0)
                        status_text = "RESTING"
                    else:
                        ms_left = int((TAP_DURATION_LIMIT - time_held) * 1000)
                        color_draw  = active_color
                        status_text = f"HOLDING ({ms_left}ms left)"
            else:
                if key_is_down[idx]:
                    keyboard.release(target_key)
                    key_is_down[idx] = False
                    print(f"  [x] KEY UP (Clear): {reg['key'].upper()}")

        # ── Calibration overlay (always shown when in fg mode) ─────────────────
        if calibration_state == 'fg_waiting':
            color_draw  = (0, 255, 255)
            status_text = f"STEP ON PAD → press '{reg['key'].upper()}'"

        status_text += f" | Pres:{int(max_box_presence)}% | Vel:{int(max_smooth_velocity)}/s"

        cv2.polylines(frame, [pts], True, color_draw, 2)
        tx = int(reg['corners'][0][0] * width)
        ty = int(reg['corners'][0][1] * height) - 10
        cv2.putText(frame, status_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_draw, 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    # ── Keyboard input ─────────────────────────────────────────────────────────
    key = cv2.waitKey(1)

    if key == 27:   # ESC — quit
        break

    elif key in (ord('b'), ord('B')):
        # ── BACKGROUND calibration ─────────────────────────────────────────────
        # Use the CURRENT hsv_frame (captured this loop iteration) so the
        # sample is perfectly synchronised with what the user sees right now.
        print("\n━━━  BACKGROUND CALIBRATION  ━━━")
        print("  Sampling all regions (make sure the pad is clear)…")
        success_count = 0
        for reg in regions_data:
            result = sample_region_hsv(hsv_frame, reg, width, height)
            if result:
                reg['background_hsv'] = result
                print(f"  ✔  [{reg['key'].upper()}]  background HSV = {result}")
                success_count += 1
            else:
                print(f"  ✘  [{reg['key'].upper()}]  region mask was empty — check corner placement.")
        print(f"\n  {success_count}/{len(regions_data)} regions updated.")
        print("  Next: press 'f', step onto a pad and press its arrow key.\n")
        calibration_status_summary()

    elif key in (ord('f'), ord('F')):
        # ── Start FOREGROUND calibration ───────────────────────────────────────
        calibration_state   = 'fg_waiting'
        fg_calibration_mode = True
        print("\n━━━  FOREGROUND CALIBRATION  ━━━")
        print("  Step ONTO a pad, then press the matching key:")
        for reg in regions_data:
            bg_tag = "(bg ✔)" if reg.get('background_hsv') else "(bg ✘ — run 'b' first!)"
            fg_tag = "(fg already set)" if reg.get('foreground_hsv') else ""
            print(f"    [{reg['key'].upper()}]  {bg_tag} {fg_tag}")
        print("  Press any mapped key to capture, or ESC to cancel.\n")

    elif key in (ord('s'), ord('S')):
        # ── Save ───────────────────────────────────────────────────────────────
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("  💾  Config saved →", CONFIG_FILE)
        calibration_status_summary()

    elif calibration_state == 'fg_waiting' and key != -1:
        # ── Capture FOREGROUND sample for the pressed key ──────────────────────
        pressed_key_name = parse_opencv_key(key)

        if key == 27:   # ESC cancels foreground calibration
            calibration_state   = 'idle'
            fg_calibration_mode = False
            print("  ✘  Foreground calibration cancelled.\n")

        elif pressed_key_name:
            matched = False
            for reg in regions_data:
                if reg['key'].lower() == pressed_key_name:
                    result = sample_region_hsv(hsv_frame, reg, width, height)
                    if result:
                        reg['foreground_hsv'] = result
                        print(f"  ✔  [{pressed_key_name.upper()}]  foreground HSV = {result}")
                        matched = True
                    else:
                        print(f"  ✘  [{pressed_key_name.upper()}]  mask was empty — check corners.")
                    break

            if not matched:
                print(f"  ✘  Key '{pressed_key_name.upper()}' doesn't match any region.")

            # Check if all regions are now calibrated
            all_done = all(
                reg.get('background_hsv') and reg.get('foreground_hsv')
                for reg in regions_data
            )
            if all_done:
                calibration_state   = 'idle'
                fg_calibration_mode = False
                print("\n  🎉  All regions fully calibrated! Press 's' to save.\n")
                calibration_status_summary()
            else:
                # Tell the user what's still missing
                remaining = [
                    reg['key'].upper() for reg in regions_data
                    if not (reg.get('background_hsv') and reg.get('foreground_hsv'))
                ]
                print(f"  Still waiting for: {', '.join(remaining)}")
                print("  Step onto the next pad and press its key, or press 'f' to restart.\n")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
for idx, pressed in enumerate(key_is_down):
    if pressed:
        keyboard.release(get_key(regions_data[idx]['key']))