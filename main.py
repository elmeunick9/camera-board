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

# Load Config
if not os.path.exists(CONFIG_FILE):
    print(f"Error: {CONFIG_FILE} not found.")
    exit()

with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

cam_idx = config['camera']['device_index']
mirror = config['camera']['mirror_preview']
active_color = tuple(config['settings']['active_box_color'])
regions_data = config['regions']

# --- STATE AND HISTORY COOLDOWN MATRICES ---
# --- TRACKING MATRICES ---
kernel_history = {}          # Spatial history matrix: {(idx, kx, ky): deque}
visual_flash_timers = [0] * len(regions_data)
last_trigger_times = [0.0] * len(regions_data)

# Clear separation of physical reality vs digital emulation
foot_present = [False] * len(regions_data) # Physical sensor state
key_is_down = [False] * len(regions_data)  # Virtual OS keyboard state

selected_region_idx = -1
selected_corner_idx = -1
fg_calibration_mode = False
last_frame_time = time.time()

def get_key(key_str):
    special_keys = {"space": Key.space, "enter": Key.enter, "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right}
    return special_keys.get(key_str.lower(), key_str)

def parse_opencv_key(key_code):
    if key_code == 81 or key_code == 2424832: return "left"
    if key_code == 82 or key_code == 2490368: return "up"
    if key_code == 83 or key_code == 2555904: return "right"
    if key_code == 84 or key_code == 2621440: return "down"
    try: return chr(key_code & 0xFF).lower()
    except ValueError: return ""

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
        nx = max(0.0, min(1.0, x / width))
        ny = max(0.0, min(1.0, y / height))
        regions_data[selected_region_idx]['corners'][selected_corner_idx] = [nx, ny]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_region_idx = -1
        selected_corner_idx = -1

cap = cv2.VideoCapture(cam_idx)
cv2.namedWindow(WINDOW_NAME)
dims = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
cv2.setMouseCallback(WINDOW_NAME, mouse_handler, dims)

print("=====================================================================")
print(" ⚡ 100ms PRO TUNED SYSTEM ACTIVE ⚡")
print("=====================================================================")

while True:
    ret, frame = cap.read()
    if not ret: break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break

    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time
    if dt <= 0: dt = 0.001

    if mirror: frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    dims['width'], dims['height'] = width, height
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for idx, reg in enumerate(regions_data):
        pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32).reshape((-1, 1, 2))
        rx, ry, rw, rh = cv2.boundingRect(pts)
        
        target_key = get_key(reg['key'])
        thresh_limit = reg.get('affinity_threshold', 50)
        
        max_box_presence = 0.0
        active_motion_detected = False
        max_smooth_velocity = -9999.0

        kernel_size = 16
        stride = 16
        
        # --- TUNED CONFIGURATION TIMERS (100ms Parameters) ---
        MOTION_VELOCITY_THRESH = 100  # Adjusted slightly higher for tighter window velocity tracking
        TAP_DURATION_LIMIT = 0.200    # Cut off key at exactly 100ms if body is motionless
        HISTORY_WINDOW_SEC = 0.100    # Accumulate data over a rolling 100ms window
        
        if reg.get('background_hsv') and reg.get('foreground_hsv'):
            bg = np.array(reg['background_hsv'])
            fg = np.array(reg['foreground_hsv'])
            
            for ky in range(ry, ry + rh - kernel_size, stride):
                for kx in range(rx, rx + rw - kernel_size, stride):
                    if cv2.pointPolygonTest(pts, (kx + kernel_size//2, ky + kernel_size//2), False) >= 0:
                        
                        kernel_roi = hsv_frame[ky:ky+kernel_size, kx:kx+kernel_size]
                        kernel_avg = cv2.mean(kernel_roi)[:3]
                        
                        curr = np.array(kernel_avg)
                        d_bg = np.linalg.norm(curr - bg)
                        d_fg = np.linalg.norm(curr - fg)
                        tot = d_bg + d_fg
                        
                        if tot > 0:
                            k_affinity = (d_bg / tot) * 100
                            if k_affinity > max_box_presence:
                                max_box_presence = k_affinity
                            
                            # Log matrix memory frame queue
                            mem_key = (idx, kx, ky)
                            if mem_key not in kernel_history:
                                kernel_history[mem_key] = deque(maxlen=10)
                            
                            kernel_history[mem_key].append((current_time, k_affinity))
                            
                            # Evict timestamps older than 100ms
                            while len(kernel_history[mem_key]) > 1 and (current_time - kernel_history[mem_key][0][0]) > HISTORY_WINDOW_SEC:
                                kernel_history[mem_key].popleft()
                            
                            # Compute smoothed slope over the 100ms window
                            if len(kernel_history[mem_key]) > 1:
                                oldest_time, oldest_aff = kernel_history[mem_key][0]
                                time_span = current_time - oldest_time
                                smooth_k_velocity = abs(k_affinity - oldest_aff) / (time_span if time_span > 0 else 0.001)
                            else:
                                smooth_k_velocity = 0.0
                            
                            if smooth_k_velocity > max_smooth_velocity:
                                max_smooth_velocity = smooth_k_velocity

                            if k_affinity >= thresh_limit and smooth_k_velocity > MOTION_VELOCITY_THRESH:
                                active_motion_detected = True

            # --- DYNAMIC 100ms STATE MACHINE ---
            # --- DUAL-VARIABLE HIGH-PRECISION STATE ENGINE ---
            color_draw = (0, 255, 0) # Default: Bright Green (Idle)
            status_text = "IDLE"

            # 1. Determine physical presence in this exact frame
            is_foot_here_now = (max_box_presence >= thresh_limit)

            # 2. STATE TRANSITION: Detect the exact moment of impact
            # If a foot wasn't here before, but is here now, it's a fresh stomp.
            if is_foot_here_now and not foot_present[idx]:
                last_trigger_times[idx] = current_time  # Reset clock to 0 relative to now
                
            # Update our master physical sensor tracking array for the frame
            foot_present[idx] = is_foot_here_now
            
            # Recalculate true elapsed time safely
            time_held = current_time - last_trigger_times[idx]

            # 3. --- PROCESS ACTIONS BASED ON RE-ALIGNED TIME ---
            if foot_present[idx]:
                
                if not key_is_down[idx] and time_held >= TAP_DURATION_LIMIT:
                    # Your foot has sat perfectly still past the 350ms cutoff mark.
                    color_draw = (0, 100, 0) # Dark Moss Green
                    status_text = "RESTING"
                    
                elif not key_is_down[idx] and time_held < TAP_DURATION_LIMIT:
                    # Fresh initial downstroke event (Fires exactly once on entry)
                    keyboard.press(target_key)
                    key_is_down[idx] = True
                    print(f"[-] KEY DOWN (Fresh Step): {reg['key'].upper()}")
                    color_draw = active_color # Pink
                    status_text = "TAP TRIGGER"
                    
                elif active_motion_detected:
                    # Continuous movement shuffles and extends the hold countdown
                    last_trigger_times[idx] = current_time 
                    color_draw = (255, 255, 0) # Cyan
                    status_text = "HOLD SUSTAINED"
                    
                    if not key_is_down[idx]:
                        keyboard.press(target_key)
                        key_is_down[idx] = True
                        print(f"[-] KEY DOWN (Hold Resumed via Motion): {reg['key'].upper()}")
                
                else:
                    # Foot is present but stationary, waiting out the remaining 350ms window
                    if time_held >= TAP_DURATION_LIMIT:
                        if key_is_down[idx]:
                            keyboard.release(target_key)
                            key_is_down[idx] = False
                            print(f"[x] KEY UP (Auto-Released): {reg['key'].upper()}")
                        color_draw = (0, 100, 0) # Dark Moss Green
                        status_text = "RESTING"
                    else:
                        color_draw = active_color # Pink
                        status_text = f"HOLDING ({int((TAP_DURATION_LIMIT - time_held)*1000)}ms left)"
            
            else:
                # --- FOOT IS COMPLETELY REMOVED FROM THE REGION ---
                if key_is_down[idx]:
                    keyboard.release(target_key)
                    key_is_down[idx] = False
                    print(f"[x] KEY UP (Clear): {reg['key'].upper()}")

            if fg_calibration_mode:
                color_draw = (0, 255, 255)
                status_text = "CALIBRATION LISTENING"

            status_text += f" | Pres: {int(max_box_presence)}% | Vel: {int(max_smooth_velocity)}/s"
                    
        else:
            color_draw = (0, 165, 255)
            status_text = "UNCALIBRATED - Press 'b' then 'f'"

        cv2.polylines(frame, [pts], True, color_draw, 2)
        text_pos = (int(reg['corners'][0][0] * width), int(reg['corners'][0][1] * height) - 10)
        cv2.putText(frame, status_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_draw, 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)
    
    # Process keyboard commands
    key = cv2.waitKey(1)
    if key == 27: # ESC
        break
    elif key == ord('b') or key == ord('B'): 
        for reg in regions_data:
            pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32)
            mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            reg['background_hsv'] = [int(x) for x in cv2.mean(hsv_frame, mask=mask)[:3]]
        print("✔ Background floor profile updated.")
    elif key == ord('f') or key == ord('F'):
        fg_calibration_mode = True
        print("🟨 Foreground mode active. Step in a box and hit its matching arrow key...")
    elif fg_calibration_mode and key != -1:
        pressed_key_name = parse_opencv_key(key)
        matched = False
        for reg in regions_data:
            if reg['key'].lower() == pressed_key_name:
                pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32)
                mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                reg['foreground_hsv'] = [int(x) for x in cv2.mean(hsv_frame, mask=mask)[:3]]
                print(f"✔ Linked foreground signature to game action: '{pressed_key_name.upper()}'")
                matched = True
                break
        fg_calibration_mode = False
        if not matched and pressed_key_name != "":
            print(f"❌ Key '{pressed_key_name.upper()}' didn't match any region. Mode closed.")
    elif key == ord('s') or key == ord('S'):
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("💾 Configuration file safely saved!")

cap.release()
cv2.destroyAllWindows()
for idx, pressed in enumerate(region_pressed):
    if pressed: keyboard.release(get_key(regions_data[idx]['key']))