# -*- coding: utf-8 -*-
"""
人脸比对脚本2：比较一张人脸与人群照片中的匹配
Usage: python face_compare_one_to_many.py --query path/to/query.jpg --gallery path/to/group.jpg
"""

import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def compare_one_to_many(query_path, gallery_path):
    """在一张群像中找到与目标人脸最相似的人"""
    print(f"查询人脸: {query_path}")
    print(f"人群照片: {gallery_path}")

    # 初始化模型
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 用 cv2 读取图片并转为 RGB
    query_img = cv2.imread(query_path)
    gallery_img = cv2.imread(gallery_path)
    if query_img is None or gallery_img is None:
        print("✗ 错误: 无法读取图片")
        return None

    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    gallery_img_rgb = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

    # 提取查询人脸特征
    print("\n正在提取查询人脸特征...")
    query_faces = app.get(query_img_rgb)
    if len(query_faces) == 0:
        print("✗ 未在查询图片中检测到人脸")
        return None

    query_feat = query_faces[0].embedding
    print(f"查询特征维度: {query_feat.shape}")

    # 提取人群中所有人脸特征
    print("\n正在检测人群中的人脸...")
    gallery_faces = app.get(gallery_img_rgb)

    if len(gallery_faces) == 0:
        print("✗ 未在人群照片中检测到人脸")
        return None

    print(f"检测到 {len(gallery_faces)} 张人脸")

    # 计算相似度
    similarities = []
    for i, face in enumerate(gallery_faces):
        sim = np.dot(query_feat, face.embedding) / (np.linalg.norm(query_feat) * np.linalg.norm(face.embedding))
        det_score = face.get('det_score', 0)
        similarities.append((i, sim, det_score))
        print(f"  人脸 {i+1}: 相似度={sim:.4f}, 检测置信度={det_score:.2f}")

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*50}")
    print(f"最相似人脸: 人脸 {similarities[0][0]+1}")
    print(f"相似度: {similarities[0][1]:.4f}")
    print(f"{'='*50}")

    if similarities[0][1] > 0.5:
        print("✓ 判断: 人群中包含同一人")
    elif similarities[0][1] > 0.35:
        print("○ 判断: 可能是同一人 (需要更多证据)")
    else:
        print("✗ 判断: 人群中不含此人")

    return similarities

def main():
    parser = argparse.ArgumentParser(description='在一张群像中找到与目标人脸最相似的人')
    parser.add_argument('--query', type=str, required=True, help='查询人脸图片路径')
    parser.add_argument('--gallery', type=str, required=True, help='人群照片路径')
    args = parser.parse_args()

    print("=" * 50)
    print("人脸搜索比对")
    print("=" * 50)
    compare_one_to_many(args.query, args.gallery)

if __name__ == '__main__':
    main()