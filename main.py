import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# 1. Spawn a visual UI file picker
print("Awaiting file selection...")
root = tk.Tk()
root.withdraw()           
root.attributes('-topmost', True) 

input_path = filedialog.askopenfilename(
    title="Upload Drone Surveillance Footage",
    filetypes=[("Media Files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png")]
)

if not input_path:
    print("No file selected. Aborting mission.")
    exit()

print(f"Target acquired: {input_path}")

# 2. Load your custom Drone AI
print("Loading custom AI brain...")
model = YOLO("best.pt")

# ==========================================
# --- UPGRADED DYNAMIC STYLING FUNCTION ---
# ==========================================
def apply_tactical_styling(img, results, is_video=False):
    result = results[0]
    
    if result.boxes is None:
        return img
        
    # Measure the canvas to calculate dynamic text sizes
    img_h, img_w = img.shape[:2]
    base_scale = min(img_w, img_h) / 1000.0  
        
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # --- DYNAMIC SIZE ADJUSTMENTS ---
        if is_video:
            if box.id is not None:
                track_id = int(box.id[0])
                label = f"{class_name} {track_id}"
            else:
                label = f"{class_name}"
            
            # VIDEO SETTINGS: Untouched, exactly how you liked them
            t_scale = max(0.6, base_scale * 1.2)            
            t_thickness = max(2, int(t_scale * 2))          
            box_thickness = max(2, int(t_scale * 1.5))
            
        else:
            conf = float(box.conf[0])
            label = f"{class_name} {conf:.2f}"
            
            # IMAGE SETTINGS: Crushed down to exactly 3/5ths of the previous dynamic size
            t_scale = max(0.24, base_scale * 0.48)            
            t_thickness = max(1, int(t_scale * 1.2))          
            box_thickness = max(1, int(t_scale * 1.2))

        color = (0, 255, 0)  # Pure Bright Green
        t_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=box_thickness)

        t_size = cv2.getTextSize(label, t_font, t_scale, t_thickness)[0]
        c2 = x1 + t_size[0] + 1, y1 - t_size[1] - 1 

        cv2.rectangle(img, (x1, y1), c2, color, -1) 
        cv2.putText(img, label, (x1, y1 - 1), t_font, t_scale, (0, 0, 0), thickness=t_thickness)
        
    return img

# ==========================================
# --- AUTO-DETECT IMAGE OR VIDEO ---
# ==========================================
ext = os.path.splitext(input_path)[1].lower()
image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
video_exts = ['.mp4', '.avi', '.mov', '.mkv']

# Get the filename without the full path for saving cleanly
file_name = os.path.basename(input_path)

if ext in image_exts:
    print(f"Processing Image: {file_name}")
    img = cv2.imread(input_path)
    results = model.predict(source=img, save=False, show=False, verbose=False)
    img = apply_tactical_styling(img, results, is_video=False)
    
    output_name = f"tactical_{file_name}"
    cv2.imwrite(output_name, img)
    print(f"Operational view saved as: {output_name}")

elif ext in video_exts:
    print(f"Processing Video: {file_name}")
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_name = f"tactical_{file_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    print("Tracker Activated. Let the i3 cook...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}/{total_frames}...")
            
        results = model.track(source=frame, persist=True, save=False, show=False, verbose=False)
        frame = apply_tactical_styling(frame, results, is_video=True) 
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Operational video saved as: {output_name}")

else:
    print("Unsupported file type! Throw a standard image or video at it.")