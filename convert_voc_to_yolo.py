# -*- coding: utf-8 -*-
"""
Convert Pascal VOC XML annotations to YOLO .txt format
Usage: python convert_voc_to_yolo.py --voc-root D:\VOCdevkit\VOC2028 --output dataset
"""

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random


CLASSES = ['hat', 'person']  # class 0=hat, class 1=person


def parse_voc_xml(xml_path):
    """Parse VOC XML file and return image size and bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    bboxes = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES.index(cls_name)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bboxes.append((cls_id, xmin, ymin, xmax, ymax))

    return img_w, img_h, bboxes


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert VOC bbox (xmin,ymin,xmax,ymax) to YOLO format (center_x, center_y, w, h) normalized"""
    center_x = (xmin + xmax) / 2 / img_w
    center_y = (ymin + ymax) / 2 / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h
    return center_x, center_y, box_w, box_h


def process_dataset(voc_root, output_root, train_split=0.8, seed=42):
    """
    Convert VOC dataset to YOLO format

    Args:
        voc_root: Path to VOC dataset root (e.g., D:\VOCdevkit\VOC2028)
        output_root: Path to output directory (e.g., dataset)
        train_split: Train/val split ratio (0.8 = 80% train, 20% val)
        seed: Random seed for reproducibility
    """
    voc_root = Path(voc_root)
    output_root = Path(output_root)

    # Directories
    images_dir = output_root / 'images'
    labels_dir = output_root / 'labels'

    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # Check VOC structure
    annotations_dir = voc_root / 'Annotations'
    jpeg_images_dir = voc_root / 'JPEGImages'
    imagesets_dir = voc_root / 'ImageSets' / 'Main'

    if not annotations_dir.exists():
        raise ValueError(f"Annotations dir not found: {annotations_dir}")
    if not jpeg_images_dir.exists():
        raise ValueError(f"JPEGImages dir not found: {jpeg_images_dir}")

    # Get all XML files
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} XML annotation files")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * train_split)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split_name} set ({len(files)} files)...")

        for i, xml_path in enumerate(files):
            if i % 500 == 0 and i > 0:
                print(f"  Processed {i}/{len(files)}")

            # Parse XML
            try:
                img_w, img_h, bboxes = parse_voc_xml(xml_path)
            except Exception as e:
                print(f"  Warning: Failed to parse {xml_path}: {e}")
                continue

            # Get image file
            img_name = xml_path.stem + '.jpg'
            img_src = jpeg_images_dir / img_name
            if not img_src.exists():
                img_src = jpeg_images_dir / (xml_path.stem + '.png')
            if not img_src.exists():
                print(f"  Warning: Image not found for {xml_path.stem}")
                continue

            # Copy image
            img_dst = images_dir / split_name / img_name
            if not img_dst.exists():
                shutil.copy2(img_src, img_dst)

            # Write label file
            label_path = labels_dir / split_name / (xml_path.stem + '.txt')
            with open(label_path, 'w') as f:
                for cls_id, xmin, ymin, xmax, ymax in bboxes:
                    cx, cy, w, h = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"\nConversion complete!")
    print(f"Images: {output_root / 'images'}")
    print(f"Labels: {output_root / 'labels'}")

    # Generate data.yaml
    data_yaml = {
        'path': str(output_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }

    yaml_path = output_root.parent / 'data.yaml'
    with open(yaml_path, 'w') as f:
        for key, value in data_yaml.items():
            f.write(f"{key}: {value}\n")

    print(f"\nGenerated data.yaml at: {yaml_path}")
    return str(yaml_path)


def generate_data_yaml(dataset_path, classes, yaml_path='data.yaml'):
    """Generate data.yaml for Ultralytics YOLO training"""
    with open(yaml_path, 'w') as f:
        f.write(f"# Safety Helmet Wearing Dataset\n")
        f.write(f"# Generated by convert_voc_to_yolo.py\n\n")
        f.write(f"path: {dataset_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names:\n")
        for i, cls in enumerate(classes):
            f.write(f"  {i}: {cls}\n")
    print(f"Generated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert VOC annotations to YOLO format')
    parser.add_argument('--voc-root', type=str, required=True,
                        help='Path to VOC dataset root (e.g., D:\\VOCdevkit\\VOC2028)')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    args = parser.parse_args()

    print(f"VOC Root: {args.voc_root}")
    print(f"Output: {args.output}")
    print(f"Train Split: {args.train_split}")
    print("-" * 50)

    process_dataset(args.voc_root, args.output, args.train_split)

    # Generate data.yaml in project root
    project_root = Path(__file__).parent
    generate_data_yaml(str(project_root / args.output), CLASSES, str(project_root / 'data.yaml'))

    print("\nDone! Next steps:")
    print("1. Copy dataset/ and data.yaml to your 3060 PC")
    print("2. Run train_yolov8.py on the 3060 PC")
    print("3. Copy best.pt back to this PC for inference")


if __name__ == '__main__':
    main()