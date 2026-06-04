# -*- coding: utf-8 -*-
"""
人脸比对脚本1：比较2张头像的相似度
Usage: python face_compare_two.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
"""

import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def compare_two(img1_path, img2_path):
    """比较两张人脸的相似度"""
    print(f"图像1: {img1_path}")
    print(f"图像2: {img2_path}")

    # 初始化模型
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 用 cv2 读取图片并转为 RGB
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("✗ 错误: 无法读取图片")
        return None

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 构建人脸特征
    print("\n正在提取特征...")
    faces1 = app.get(img1_rgb)
    faces2 = app.get(img2_rgb)

    if len(faces1) == 0 or len(faces2) == 0:
        print("✗ 错误: 未检测到人脸")
        return None

    feat1 = faces1[0].embedding
    feat2 = faces2[0].embedding

    # 计算余弦相似度
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    print(f"\n{'='*50}")
    print(f"相似度: {sim:.4f}")
    print(f"{'='*50}")

    if sim > 0.5:
        print("✓ 判断: 同一人 (可能性较高)")
    elif sim > 0.35:
        print("○ 判断: 可能是同一人 (需要更多证据)")
    else:
        print("✗ 判断: 不同人")

    return sim

def main():
    parser = argparse.ArgumentParser(description='比较2张人脸的相似度')
    parser.add_argument('--img1', type=str, required=True, help='第一张图片路径')
    parser.add_argument('--img2', type=str, required=True, help='第二张图片路径')
    args = parser.parse_args()

    print("=" * 50)
    print("人脸相似度比对")
    print("=" * 50)
    compare_two(args.img1, args.img2)

if __name__ == '__main__':
    main()