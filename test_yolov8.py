# -*- coding: utf-8 -*-
"""
YOLOv8 Image Detection for Safety Helmet Wearing Dataset
Reference: test_yolo.py (GluonCV version)
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


CLASSES = ['hat', 'person']
COLORS = [(0, 0, 255), (255, 0, 0)]  # hat=red, person=blue (BGR)


def detect_and_draw(img, model, conf_threshold=0.4, imgsz=416):
    """Run detection and draw bounding boxes on image"""
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


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 image detection')
    parser.add_argument('--model', type=str, default='runs/detect/shwd/weights/best.pt',
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Input image size')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    args = parser.parse_args()

    # Load model
    device = 'cpu' if args.cpu else '0'
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot read image {args.image}")
        return

    # Detect and draw
    result_img = detect_and_draw(img, model, args.conf, args.imgsz)

    # Save result
    input_path = Path(args.image)
    output_path = input_path.parent / f"{input_path.stem}_result{input_path.suffix}"
    cv2.imwrite(str(output_path), result_img)
    print(f"Saved result to: {output_path}")

    # Display
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()