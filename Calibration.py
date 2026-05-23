import cv2
import json
import numpy as np
from ultralytics import YOLO
import mss
import time
import tkinter as tk
from threading import Thread

# ---------------- CONFIG ----------------
MODEL_PATH = "composite.pt"
OUT_JSON   = "ROI_final_new.json"
OLD_JSON   = "regions.json"  # old regions to overlay
CONF_THRES = 0.5
# ---------------------------------------

model = YOLO(MODEL_PATH)
model.to('cpu')
print("Model loaded")

last_img = None
last_regions = None
status_box = None

def set_status(text):
    if status_box:
        status_box.after(0, lambda: (
            status_box.delete("1.0", tk.END),
            status_box.insert(tk.END, text)
        ))

def take_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = np.array(sct.grab(monitor))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

def norm_box(x1, y1, x2, y2, w, h):
    return {"x1": float(x1/w), "y1": float(y1/h), "x2": float(x2/w), "y2": float(y2/h)}

def split_columns(box, n, w, h):
    x1, y1, x2, y2 = box
    cols = []
    step = (x2 - x1)/n
    for i in range(n):
        cx1 = x1 + i*step
        cx2 = cx1 + step
        cols.append(norm_box(cx1, y1, cx2, y2, w, h))
    return cols

def calibrate():
    global last_img, last_regions
    set_status("Calibrating...\nTaking screenshot and running detection...")

    try:
        img = take_screenshot()
        img = cv2.imread("Screenshot 2026-02-04 181230.png")
        H, W = img.shape[:2]

        res = model(img, conf=CONF_THRES, verbose=False)[0]

        found = {}
        for b in res.boxes:
            cls = int(b.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            found[name] = (x1, y1, x2, y2)
            if name == "stock":
                tabx2 = x2
                wdiff = x2 - x1
                whi = y1
            if name == "foundation":
                tabx1 = x1

        for b in res.boxes:
            cls = int(b.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            if name == "tablaue":
                found[name] = (tabx1, y1, tabx2, int(y1 + ((y2 - y1)*1.154)))
            elif name == "waste":
                found[name] = (x2 - wdiff, whi, x2, y2)
            else:
                found[name] = (x1, y1, x2, y2)

        required = {"foundation", "stock", "tablaue", "waste"}
        if not required.issubset(found):
            missing = required - set(found.keys())
            msg = f"Calibration failed.\nMissing: {missing}"
            print(msg)
            set_status(msg)
            return

        regions = {}
        # Foundation
        f_cols = split_columns(found["foundation"], 4, W, H)
        for i, r in enumerate(f_cols):
            regions[f"f{i}"] = r
        # Tableau
        t_cols = split_columns(found["tablaue"], 7, W, H)
        for i, r in enumerate(t_cols):
            regions[f"t{i}"] = r
        # Stock & Waste
        sx1, sy1, sx2, sy2 = found["stock"]
        wx1, wy1, wx2, wy2 = found["waste"]
        regions["stock"] = norm_box(sx1, sy1, sx2, sy2, W, H)
        regions["waste"] = norm_box(wx1, wy1, wx2, wy2, W, H)

        with open(OUT_JSON, "w") as f:
            json.dump(regions, f, indent=2)

        last_img = img
        last_regions = regions
        msg = "Calibration completed successfully.\nROI.json saved."
        print(msg)
        set_status(msg)

    except Exception as e:
        err = f"Calibration error:\n{str(e)}"
        print(err)
        set_status(err)

def show_roi():
    if last_img is None or last_regions is None:
        set_status("Run Calibrate first.")
        return

    vis = last_img.copy()
    H, W = vis.shape[:2]

    # ---------------- Draw RO.json regions first (green) ----------------
    for name, r in last_regions.items():
        x1 = int(r["x1"] * W)
        y1 = int(r["y1"] * H)
        x2 = int(r["x2"] * W)
        y2 = int(r["y2"] * H)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, name, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---------------- Draw regions_595.json boxes on top (blue) ----------------
    try:
        with open(OLD_JSON, "r") as f:
            old_regions = json.load(f)

        # Determine the bounding box covering f0..t6
        f0 = last_regions["f0"]
        t6 = last_regions["t6"]
        base_x1 = f0["x1"] * W
        base_y1 = f0["y1"] * H
        base_x2 = t6["x2"] * W
        base_y2 = t6["y2"] * H
        base_w = base_x2 - base_x1
        base_h = base_y2 - base_y1

        for name, r in old_regions.items():
            # Map normalized 0-1 from regions_595 to absolute image coordinates
            x1 = int(base_x1 + r["x1"] * base_w)
            y1 = int(base_y1 + r["y1"] * base_h)
            x2 = int(base_x1 + r["x2"] * base_w)
            y2 = int(base_y1 + r["y2"] * base_h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, name, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    except Exception as e:
        print(f"Could not load {OLD_JSON}: {e}")

    cv2.imshow("Game Region", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_calibrate_thread():
    Thread(target=calibrate, daemon=True).start()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Solitaire ROI Calibrator")
root.geometry("360x250")

btn_cal = tk.Button(root, text="Calibrate", width=25, command=run_calibrate_thread)
btn_cal.pack(pady=8)

btn_show = tk.Button(root, text="Show ROI", width=25, command=show_roi)
btn_show.pack(pady=5)

status_box = tk.Text(root, height=5, width=42)
status_box.pack(pady=10)
status_box.insert(tk.END, "Press Calibrate")

root.mainloop()
