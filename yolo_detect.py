import os
import sys
import argparse
import glob
import time
import csv

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--count-mode', help='Counting mode: "line" = count only when crossing a drawn line, "confidence" = count detections by confidence threshold.',
                    choices=['line', 'confidence', 'ask'],
                    default='ask')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
# Allow passing threshold as percent (e.g. 80) or fraction (e.g. 0.8)
if min_thresh > 1.0:
    min_thresh = min_thresh / 100.0
min_thresh = max(0.0, min(1.0, min_thresh))
user_res = args.resolution
record = args.record
count_mode = args.count_mode

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

single_class_name = 'sack'

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Ask user for counting mode at startup (unless explicitly set)
if count_mode == 'ask':
    print('\nSelect counting mode:')
    print('  1) Draw a counting line (count when object crosses the line)')
    print('  2) Confidence counting (count detections above threshold)')
    choice = input('Enter 1 or 2 [1]: ').strip()
    if choice == '2':
        count_mode = 'confidence'
    else:
        count_mode = 'line'

# Line counting needs a stream (video/camera). If source is a single image/folder, fall back.
if count_mode == 'line' and source_type in ['image', 'folder']:
    print('NOTE: Line-crossing counting is only supported for video/camera streams. Falling back to confidence counting.')
    count_mode = 'confidence'

# Threshold rules:
# - display_thresh controls what boxes are shown
# - count_thresh controls what detections are counted
display_thresh = min_thresh
count_thresh = min_thresh
if count_mode == 'confidence':
    count_thresh = max(min_thresh, 0.60)
    if count_thresh > display_thresh:
        print(f'NOTE: Confidence mode will DISPLAY >= {display_thresh:.2f} but COUNT only >= {count_thresh:.2f}.')

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Detection statistics
total_object_count = 0
frame_index = 0
detection_history = []  # per-frame stats
start_time = time.time()
max_objects_in_frame = 0
peak_time_sec = None

# Line-crossing counting setup
counting_line = []  # will hold two points [(x1, y1), (x2, y2)]
line_set = False
crossing_count = 0


def _mouse_callback(event, x, y, flags, param):
    global counting_line, line_set
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(counting_line) < 2:
            counting_line.append((x, y))
        if len(counting_line) == 2:
            line_set = True


def _setup_counting_line(frame):
    """
    Let the user draw a custom counting line on the first frame
    by clicking two points. Works per-video.
    """
    global counting_line, line_set
    counting_line = []
    line_set = False

    clone = frame.copy()
    cv2.namedWindow('Set Counting Line')
    cv2.setMouseCallback('Set Counting Line', _mouse_callback)

    while True:
        display = clone.copy()
        if len(counting_line) >= 1:
            cv2.circle(display, counting_line[0], 5, (0, 255, 0), -1)
        if len(counting_line) == 2:
            cv2.circle(display, counting_line[1], 5, (0, 255, 0), -1)
            cv2.line(display, counting_line[0], counting_line[1], (0, 255, 255), 2)
            msg = 'Press ENTER to confirm'
            (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (6, 6), (6 + tw + 12, 6 + th + base + 12), (0, 0, 0), -1)
            cv2.putText(display, msg, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            msg = 'Click two points to set line'
            (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (6, 6), (6 + tw + 12, 6 + th + base + 12), (0, 0, 0), -1)
            cv2.putText(display, msg, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Set Counting Line', display)
        key = cv2.waitKey(20)
        # ENTER or RETURN key to confirm
        if (key == 13 or key == 10) and line_set and len(counting_line) == 2:
            break
        if key == 27:  # ESC to cancel
            counting_line = []
            break

    cv2.destroyWindow('Set Counting Line')


def _point_side_of_line(pt, p1, p2):
    """
    Returns positive / negative / zero indicating which side of the line the point is on.
    """
    x, y = pt
    x1, y1 = p1
    x2, y2 = p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)



tracks = {}
next_track_id = 0
max_track_lost = 10

# Confidence-mode unique counting (simple tracking)
conf_tracks = {}
conf_next_track_id = 0
conf_total_count = 0
min_conf_track_age = 3

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # On the first frame, let user set a custom counting line (line mode only)
    if count_mode == 'line' and frame_index == 0:
        _setup_counting_line(frame)

    # Run inference on frame
    results = model(frame, verbose=False)
    
    # Extract results
    detections = results[0].boxes
    
    # Initialize variable for basic object counting example
    object_count = 0
    frame_class_counts = {}
    
    # Data for simple tracking
    detections_info = []
    
    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
    
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
    
        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
    
        # Get bounding box confidence
        conf = detections[i].conf.item()
    
        # Draw box if confidence is high enough to display
        if conf >= display_thresh:
    
            color = bbox_colors[classidx % 10]
            thickness = 2
            if conf >= count_thresh:
                thickness = 3  # make "count-eligible" detections stand out
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
    
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
            # Single-class model: count only if confidence meets count threshold
            if conf >= count_thresh:
                detections_info.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'center': (cx, cy),
                    'classname': single_class_name
                })

    if count_mode == 'line':
        # Centroid-based tracking for line crossing (simple + forgiving)
        new_tracks = {}
        used_track_ids = set()
        distance_threshold = 50  # pixels

        for det in detections_info:
            cx, cy = det['center']
            best_track_id = None
            best_dist = None
            for tid, tr in tracks.items():
                tx, ty = tr['center']
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None and best_dist is not None and best_dist < distance_threshold:
                track = tracks[best_track_id]
                prev_side = track['side']
                if len(counting_line) == 2:
                    current_side = _point_side_of_line((cx, cy), counting_line[0], counting_line[1])
                    if prev_side is not None and current_side is not None and prev_side * current_side < 0:
                        crossing_count += 1
                        total_object_count += 1
                    track['side'] = current_side
                track['center'] = (cx, cy)
                track['lost'] = 0
                new_tracks[best_track_id] = track
                used_track_ids.add(best_track_id)
            else:
                side_val = None
                if len(counting_line) == 2:
                    side_val = _point_side_of_line((cx, cy), counting_line[0], counting_line[1])
                new_tracks[next_track_id] = {
                    'center': (cx, cy),
                    'side': side_val,
                    'lost': 0
                }
                next_track_id += 1

        # Age existing tracks that were not matched
        for tid, tr in tracks.items():
            if tid in new_tracks:
                continue
            tr['lost'] += 1
            if tr['lost'] <= max_track_lost:
                new_tracks[tid] = tr

        tracks = new_tracks

        object_count = len(detections_info)
    else:
        # Confidence mode: track detections and count each physical sack once
        object_count = len(detections_info)
        new_conf_tracks = {}
        used_conf_track_ids = set()
        conf_distance_threshold = 60  # pixels

        for det in detections_info:
            cx, cy = det['center']
            best_track_id = None
            best_dist = None
            for tid, tr in conf_tracks.items():
                tx, ty = tr['center']
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None and best_dist is not None and best_dist < conf_distance_threshold:
                track = conf_tracks[best_track_id]
                track['center'] = (cx, cy)
                track['lost'] = 0
                track['age'] = track.get('age', 0) + 1
                # Count this track once it has been stable for a few frames
                if not track.get('counted', False) and track['age'] >= min_conf_track_age:
                    conf_total_count += 1
                    track['counted'] = True
                new_conf_tracks[best_track_id] = track
                used_conf_track_ids.add(best_track_id)
            else:
                # New track candidate – do not count immediately
                new_conf_tracks[conf_next_track_id] = {
                    'center': (cx, cy),
                    'lost': 0,
                    'age': 1,
                    'counted': False
                }
                conf_next_track_id += 1

        for tid, tr in conf_tracks.items():
            if tid in new_conf_tracks:
                continue
            tr['lost'] += 1
            if tr['lost'] <= max_track_lost:
                new_conf_tracks[tid] = tr

        conf_tracks = new_conf_tracks

    elapsed = time.time() - start_time

    if object_count > max_objects_in_frame:
        max_objects_in_frame = object_count
        peak_time_sec = elapsed
    
    detection_history.append({
        'timestamp_sec': elapsed,
        'object_count': object_count
    })
    frame_index += 1
    
    # Draw counting line if set (line mode)
    overlay_height = 90
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
    if count_mode == 'line' and len(counting_line) == 2:
        cv2.line(panel, counting_line[0], counting_line[1], (0, 255, 255), 2)
    alpha = 0.6
    frame = cv2.addWeighted(panel, alpha, frame, 1 - alpha, 0)
    
    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display detection results and aggregate statistics
    count_label = single_class_name
    cv2.putText(frame, f'Current {count_label}: {object_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if count_mode == 'line':
        cv2.putText(frame, f'Line crossings: {crossing_count}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f'Total counted: {conf_total_count}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('YOLO detection results', frame)
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    elif (key == ord('l') or key == ord('L')) and count_mode == 'line':  # Press 'l' to redraw line
        _setup_counting_line(frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# After loop ends, save detection statistics and show a creative summary

# Save per-frame counts to CSV
if detection_history:
    csv_filename = 'detection_log.csv'
    try:
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ['timestamp_sec', 'object_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in detection_history:
                writer.writerow(row)
        print(f'Detection log saved to "{csv_filename}".')
    except IOError as e:
        print(f'Failed to write detection log CSV: {e}')

# Build a simple summary image as a creative dashboard
summary_height = 400
summary_width = 600
summary_img = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
summary_img[:] = (20, 20, 20)

cv2.putText(summary_img, 'Detection Summary', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
final_total_count = crossing_count if count_mode == 'line' else conf_total_count
cv2.putText(summary_img, f'Total counted objects: {final_total_count}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(summary_img, f'Average FPS: {avg_frame_rate:.2f}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if max_objects_in_frame > 0:
    cv2.putText(summary_img, f'Peak objects in a frame: {max_objects_in_frame}', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if peak_time_sec is not None:
        cv2.putText(summary_img, f'Peak time (sec): {peak_time_sec:.2f}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow('Detection Summary', summary_img)
cv2.imwrite('detection_summary.png', summary_img)
cv2.waitKey(0)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
