# -*- coding: utf-8 -*-
"""
YOLOv8 Training Script for Safety Helmet Wearing Dataset
Run on PC with NVIDIA RTX 3060 (12GB VRAM)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml='data.yaml', model='yolov8s.pt', epochs=100, imgsz=640, batch=32, device=0):
    """
    Train YOLOv8 model

    Args:
        data_yaml: Path to data.yaml configuration
        model: Pretrained model (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (32 for 3060 12GB)
        device: GPU device (0 = first GPU)
    """
    # Load pretrained YOLOv8 model
    model = YOLO(model)
    print(f"Loaded model: {model}")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=50,          # Early stopping patience
        project='runs/detect',
        name='shwd',
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        verbose=True,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\nTraining complete!")
    print(f"Best model: runs/detect/shwd/weights/best.pt")
    print(f"Last model: runs/detect/shwd/weights/last.pt")

    # Validate
    metrics = model.val()
    print(f"\nValidation metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on Safety Helmet Wearing Dataset')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Pretrained model (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (32 for 3060 12GB)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device (0,1,2... or cpu)')
    args = parser.parse_args()

    # Verify data.yaml exists
    if not Path(args.data).exists():
        print(f"Error: data.yaml not found at {args.data}")
        print("Please copy data.yaml to this directory first")
        return

    print("=" * 50)
    print("YOLOv8 Training for Safety Helmet Wearing Dataset")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print("=" * 50)

    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == '__main__':
    main()