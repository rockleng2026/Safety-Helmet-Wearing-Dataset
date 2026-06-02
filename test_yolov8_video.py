# -*- coding: utf-8 -*-
"""
YOLOv8 Video Detection for Safety Helmet Wearing Dataset
Reference: test_yolo_video.py (GluonCV version)
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


CLASSES = ['hat', 'person']
COLORS = [(0, 0, 255), (255, 0, 0)]  # hat=red, person=blue (BGR)


def detect_and_draw(img, model, conf_threshold=0.4, imgsz=416):
    """Run detection and draw bounding boxes on frame"""
    results = model.predict(img, conf=conf_threshold, imgsz=imgsz, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS[cls_id]
        label = f'{CLASSES[cls_id]} {conf:.2f}'

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 4),
                     (x1 + t_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def process_video(video_path, output_path, model_path, conf=0.4, imgsz=416, cpu=False):
    """Process video file and save annotated result"""
    device = 'cpu' if cpu else '0'
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps}fps, {frame_count} frames")

    # Output writer (use XVID codec for AVI)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot open output video writer {output_path}")
        cap.release()
        return

    print("Processing video...")
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw
        annotated = detect_and_draw(frame.copy(), model, conf, imgsz)
        out.write(annotated)

        processed += 1
        if processed % 30 == 0:
            print(f"  Processed {processed}/{frame_count} frames ({100*processed/frame_count:.1f}%)", end='\r')

    cap.release()
    out.release()
    print(f"\nVideo saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 video detection')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output video')
    parser.add_argument('--model', type=str, default='runs/detect/shwd/weights/best.pt',
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Input image size')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        cpu=args.cpu
    )


if __name__ == '__main__':
    main()