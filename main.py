import cv2
import numpy as np
import yaml
import os
import math
from pynput.keyboard import Controller, Key

keyboard = Controller()
WINDOW_NAME = "DIY Motion Pad - Editor Mode"
CONFIG_FILE = "config.yaml"

# Global UI placement tracking states
selected_region_idx = -1
selected_corner_idx = -1
fg_calibration_mode = False  # State tracking variable

def get_key(key_str):
    special_keys = {"space": Key.space, "enter": Key.enter, "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right}
    return special_keys.get(key_str.lower(), key_str)

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
print("    -> Then press the actual key (e.g., 'w', 'a') while stepping inside.")
print("  • Press 's' to permanently SAVE all modifications to config.yaml.")
print("  • Press 'ESC' or click 'X' to close.")
print("=====================================================================")

while True:
    ret, frame = cap.read()
    if not ret: break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break

    if mirror: frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    dims['width'], dims['height'] = width, height
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for idx, reg in enumerate(regions_data):
        pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32).reshape((-1, 1, 2))

        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        avg_color = cv2.mean(hsv_frame, mask=mask)[:3]

        # UI Color changes to solid Yellow if the engine is listening for a key assignment
        if fg_calibration_mode:
            color_draw = (0, 255, 255) # Yellow
            status_text = f"WAITING FOR KEY: '{reg['key'].upper()}'"
        else:
            color_draw = (0, 255, 0) # Green default passive
            thresh_limit = reg.get('affinity_threshold', 50)
            status_text = f"Target: {thresh_limit}%"

        target_key = get_key(reg['key'])

        if not fg_calibration_mode and reg.get('background_hsv') and reg.get('foreground_hsv'):
            bg = np.array(reg['background_hsv'])
            fg = np.array(reg['foreground_hsv'])
            curr = np.array(avg_color)

            dist_to_bg = np.linalg.norm(curr - bg)
            dist_to_fg = np.linalg.norm(curr - fg)
            total_dist = dist_to_bg + dist_to_fg

            if total_dist > 0:
                fg_affinity = (dist_to_bg / total_dist) * 100
                thresh_limit = reg.get('affinity_threshold', 50)
                status_text = f"Fit: {int(fg_affinity)}%/{thresh_limit}%"

                if fg_affinity >= thresh_limit:
                    color_draw = active_color
                    if not region_pressed[idx]:
                        keyboard.press(target_key)
                        region_pressed[idx] = True
                else:
                    if region_pressed[idx]:
                        keyboard.release(target_key)
                        region_pressed[idx] = False
        
        cv2.polylines(frame, [pts], True, color_draw, 2)
        for pt in reg['corners']:
            cv2.circle(frame, (int(pt[0] * width), int(pt[1] * height)), 4, (255, 255, 0), -1)

        text_pos = (int(reg['corners'][0][0] * width), int(reg['corners'][0][1] * height) - 10)
        cv2.putText(frame, f"{reg['name']}: {status_text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_draw, 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # ESC
        break
    
    elif key == ord('b'): 
        for reg in regions_data:
            pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32)
            mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            reg['background_hsv'] = [int(x) for x in cv2.mean(hsv_frame, mask=mask)[:3]]
        print("✔ Background floor profile updated.")

    elif key == ord('f'):
        # Toggle the Calibration State Machine
        fg_calibration_mode = True
        print("🟨 Foreground mode active. Step in a box and hit its matching key on your keyboard...")

    elif fg_calibration_mode and key != 0xFF:
        # User pressed a key while in calibration mode
        pressed_char = chr(key).lower()
        matched = False
        
        for reg in regions_data:
            if reg['key'].lower() == pressed_char:
                pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in reg['corners']], np.int32)
                mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                
                reg['foreground_hsv'] = [int(x) for x in cv2.mean(hsv_frame, mask=mask)[:3]]
                print(f"✔ Linked foreground signature to game action: '{pressed_char.upper()}'")
                matched = True
                break
        
        # Turn off listening state machine after registering the key stroke handler
        fg_calibration_mode = False
        if not matched:
            print("❌ Pressed key didn't match any configured region key. Mode closed.")

    elif key == ord('s'):
        config['regions'] = regions_data
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=None)
        print("💾 Configuration file safely overwritten!")

cap.release()
cv2.destroyAllWindows()
for idx, pressed in enumerate(region_pressed):
    if pressed: keyboard.release(get_key(regions_data[idx]['key']))