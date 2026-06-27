import cv2
import numpy as np
import yaml
import os
import math
import time
from pynput.keyboard import Key, Controller

keyboard = Controller()
WINDOW_NAME = "DIY Motion Pad - Editor Mode"
CONFIG_FILE = "config.yaml"

selected_region_idx = -1
selected_corner_idx = -1
fg_calibration_mode = False

def get_key(key_str):
    special_keys = {
        "space": Key.space, "enter": Key.enter, 
        "up": Key.up, "down": Key.down, 
        "left": Key.left, "right": Key.right
    }
    return special_keys.get(key_str.lower(), key_str)

# Map OpenCV specific virtual key codes to string representations for comparison
def parse_opencv_key(key_code):
    # Windows/Linux standard virtual arrow keys for OpenCV
    if key_code == 81 or key_code == 2424832: return "left"
    if key_code == 82 or key_code == 2490368: return "up"
    if key_code == 83 or key_code == 2555904: return "right"
    if key_code == 84 or key_code == 2621440: return "down"
    
    # Fallback to standard alphanumeric characters
    try:
        return chr(key_code & 0xFF).lower()
    except ValueError:
        return ""

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

if not os.path.exists(CONFIG_FILE):
    print(f"Error: {CONFIG_FILE} not found.")
    exit()

with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

cam_idx = config['camera']['device_index']
mirror = config['camera']['mirror_preview']
active_color = tuple(config['settings']['active_box_color'])
regions_data = config['regions']
region_pressed = [False] * len(regions_data)

cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    print(f"Error: Could not open camera.")
    exit()

cv2.namedWindow(WINDOW_NAME)
dims = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
cv2.setMouseCallback(WINDOW_NAME, mouse_handler, dims)

print("=====================================================================")
print(" GUI CONTROLS:")
print("  • CLICK & DRAG polygon corners directly on screen.")
print("  • Press 'b' to autodetect EMPTY BACKGROUND (all boxes at once).")
print("  • Press 'f' to enter FOREGROUND CALIBRATION MODE.")
print("    -> Then press the actual Arrow Key while stepping inside.")
print("  • Press 's' to permanently SAVE all modifications to config.yaml.")
print("  • Press 'ESC' or click 'X' to close.")
print("=====================================================================")

region_pressed = [False] * len(regions_data)
visual_flash_timers = [0] * len(regions_data)

# --- STATE AND REAL-TIME COOLDOWN MATRICES ---
kernel_memory = {}          # Tracks past affinity values: {(idx, x, y): affinity}
region_pressed = [False] * len(regions_data)
visual_flash_timers = [0] * len(regions_data)

# Tracks the exact timestamp of the absolute last keyboard event fired per region
# Initialized to 0 so they are instantly ready to trigger at startup
last_event_times = [0.0] * len(regions_data)

# Tracks the exact timestamp of the absolute last keyboard event fired per region
# Initialized to 0 so they are instantly ready to trigger at startup
last_event_times = [0.0] * len(regions_data) 
last_frame_time = time.time()

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
        any_kernel_stomped = False
        max_velocity_found = -9999.0

        kernel_size = 16
        stride = 16
        VELOCITY_SPIKE_THRESH = 750 
        COOLDOWN_TIME = 0.200 # 200 milliseconds absolute time-lock window
        
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
                            
                            mem_key = (idx, kx, ky)
                            previous_k_affinity = kernel_memory.get(mem_key, 0.0)
                            
                            k_velocity = (k_affinity - previous_k_affinity) / dt
                            kernel_memory[mem_key] = k_affinity
                            
                            if k_velocity > max_velocity_found:
                                max_velocity_found = k_velocity

                            if k_affinity >= thresh_limit and k_velocity > VELOCITY_SPIKE_THRESH:
                                any_kernel_stomped = True

            # --- TIME-DEBOUNCED STATE CONTROLLER ---
            color_draw = (0, 255, 0)
            status_text = "IDLE"

            # Check how many milliseconds have ticking by since this specific box acted
            time_since_last_event = current_time - last_event_times[idx]

            if max_box_presence >= thresh_limit:
                color_draw = active_color 
                status_text = "HOLD ACTIVE"
                
                if not region_pressed[idx]:
                    # CASE A: Fresh down-stomp entry event
                    keyboard.press(target_key)
                    region_pressed[idx] = True
                    last_event_times[idx] = current_time # Start the 200ms countdown clock
                    print(f"[-] KEY DOWN (Fresh): {reg['key'].upper()}")
                    
                elif any_kernel_stomped:
                    # CASE B: Velocity spike detected inside an occupied box.
                    # CRITICAL CHECK: Has the 200ms temporal debounce window expired?
                    if time_since_last_event >= COOLDOWN_TIME:
                        keyboard.release(target_key)
                        keyboard.press(target_key)
                        
                        last_event_times[idx] = current_time # Reset the cooldown clock for the next tap
                        visual_flash_timers[idx] = 4
                        print(f"[⚡] DOUBLE-TAP PULSE: {reg['key'].upper()} | Wait Time Was: {int(time_since_last_event * 1000)}ms")
                    else:
                        # Velocity spiked, but we ignored it because it's within the 200ms structural noise window
                        status_text = "DEBOUNCE BLOCK"
            else:
                if region_pressed[idx]:
                    # CASE C: Completely clearing out of the zone
                    keyboard.release(target_key)
                    region_pressed[idx] = False
                    last_event_times[idx] = current_time # Reset timer on release to clear exit ripples
                    print(f"[x] KEY UP: {reg['key'].upper()}")

            if visual_flash_timers[idx] > 0:
                color_draw = (255, 255, 0) # Cyan
                status_text = "⚡ STOMP PULSE ⚡"
                visual_flash_timers[idx] -= 1

            status_text += f" | Pres: {int(max_box_presence)}% | TimeDelta: {int(time_since_last_event * 1000)}ms"
                    
        else:
            color_draw = (0, 165, 255)
            status_text = "UNCALIBRATED - Press 'b' then 'f'"

        cv2.polylines(frame, [pts], True, color_draw, 2)
        text_pos = (int(reg['corners'][0][0] * width), int(reg['corners'][0][1] * height) - 10)
        cv2.putText(frame, status_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_draw, 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1)
    
    # Removed the '& 0xFF' mask here because full-width virtual key integer numbers 
    # are required to identify arrow configurations correctly across platforms.
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
        print("🟨 Foreground mode active. Step in a box and hit its matching arrow key on your keyboard...")

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
            print(f"❌ Key '{pressed_key_name.upper()}' didn't match any configured region key. Mode closed.")

    elif key == ord('s') or key == ord('S'):
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("💾 Configuration file safely overwritten!")

cap.release()
cv2.destroyAllWindows()
for idx, pressed in enumerate(region_pressed):
    if pressed: keyboard.release(get_key(regions_data[idx]['key']))