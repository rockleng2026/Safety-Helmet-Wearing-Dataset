# -*- coding: utf-8 -*-
"""
人脸识别GUI界面
功能：
1. 比对2张头像的相似度
2. 比较一张人脸与人群照片中的匹配（带人脸框标注）
3. 摄像头实时人脸识别与比对
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import sys

from insightface.app import FaceAnalysis


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')

        # 初始化模型
        print("正在加载模型...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("模型加载完成")

        # 摄像头相关
        self.cap = None
        self.camera_running = False
        self.camera_thread = None

        # 当前选中的图片路径
        self.img1_path = None
        self.img2_path = None
        self.query_path = None
        self.gallery_path = None

        # 摄像头相关状态
        self.ref_face_path = None  # 基准人脸路径
        self.ref_feat = None       # 基准人脸特征

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 标题
        title_label = tk.Label(self.root, text="人脸识别系统",
                               font=("Microsoft YaHei", 24, "bold"),
                               bg='#2c3e50', fg='white', pady=10)
        title_label.pack(fill=tk.X)

        # 标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: 两张人脸比对
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="两张人脸比对")
        self.setup_tab1()

        # Tab 2: 人脸搜索
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="人脸搜索")
        self.setup_tab2()

        # Tab 3: 摄像头人脸识别
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="摄像头人脸识别")
        self.setup_tab3()

    def setup_tab1(self):
        """两张人脸比对界面"""
        main_frame = ttk.Frame(self.tab1, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 图片选择区域
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=10)

        # 图片1
        ttk.Label(select_frame, text="人脸图片1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.img1_label = ttk.Label(select_frame, text="未选择文件")
        self.img1_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(select_frame, text="选择图片1", command=self.select_img1).grid(row=0, column=2, padx=5, pady=5)

        # 图片2
        ttk.Label(select_frame, text="人脸图片2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.img2_label = ttk.Label(select_frame, text="未选择文件")
        self.img2_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(select_frame, text="选择图片2", command=self.select_img2).grid(row=1, column=2, padx=5, pady=5)

        # 图片预览区域
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.img1_preview = ttk.Label(preview_frame, text="图片1预览", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.img1_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.img2_preview = ttk.Label(preview_frame, text="图片2预览", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.img2_preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # 比对按钮
        ttk.Button(main_frame, text="开始比对", command=self.compare_two).pack(pady=10)

        # 结果显示
        self.result_label1 = ttk.Label(main_frame, text="", font=("Microsoft YaHei", 14))
        self.result_label1.pack(pady=5)

    def setup_tab2(self):
        """人脸搜索界面"""
        main_frame = ttk.Frame(self.tab2, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 图片选择区域
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=10)

        # 查询图片
        ttk.Label(select_frame, text="查询人脸:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.query_label = ttk.Label(select_frame, text="未选择文件")
        self.query_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(select_frame, text="选择查询图片", command=self.select_query).grid(row=0, column=2, padx=5, pady=5)

        # 群像图片
        ttk.Label(select_frame, text="人群照片:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.gallery_label = ttk.Label(select_frame, text="未选择文件")
        self.gallery_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(select_frame, text="选择人群照片", command=self.select_gallery).grid(row=1, column=2, padx=5, pady=5)

        # 图片预览区域
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.query_preview = ttk.Label(preview_frame, text="查询人脸预览", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.query_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.gallery_preview = ttk.Label(preview_frame, text="人群照片预览", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.gallery_preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # 搜索按钮
        ttk.Button(main_frame, text="开始搜索并标注", command=self.search_face).pack(pady=10)

        # 结果显示
        self.result_label2 = ttk.Label(main_frame, text="", font=("Microsoft YaHei", 14))
        self.result_label2.pack(pady=5)

    def setup_tab3(self):
        """摄像头人脸识别界面"""
        main_frame = ttk.Frame(self.tab3, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左右布局的容器
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧：基准人脸选择区域（限制宽度）
        left_frame = ttk.LabelFrame(content_frame, text="基准人脸 & 截图结果", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=5)
        left_frame.configure(width=300)
        left_frame.pack_propagate(False)

        # 上传区域
        ttk.Label(left_frame, text="基准人脸:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ref_label = ttk.Label(left_frame, text="未选择")
        self.ref_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(left_frame, text="上传", command=self.select_ref_face).grid(row=0, column=2, padx=5, pady=2)

        # 基准人脸预览 - 自适应图像大小
        self.ref_preview = tk.Label(left_frame, text="基准预览", relief=tk.SUNKEN, anchor=tk.CENTER,
                                     bg='#444', fg='white', font=("Microsoft YaHei", 10))
        self.ref_preview.grid(row=1, column=0, columnspan=3, pady=5, ipady=5, ipadx=5)

        # 分隔线
        ttk.Separator(left_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)

        # 截图结果显示区域 - 自适应图像大小
        ttk.Label(left_frame, text="截图结果:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.capture_result_label = tk.Label(left_frame, text="暂无", relief=tk.SUNKEN, anchor=tk.CENTER,
                                             bg='#333', fg='white', font=("Microsoft YaHei", 10))
        self.capture_result_label.grid(row=4, column=0, columnspan=3, pady=5, ipady=5, ipadx=5)

        # 基准人脸状态
        self.ref_status = ttk.Label(left_frame, text="", font=("Microsoft YaHei", 9), foreground='gray')
        self.ref_status.grid(row=5, column=0, columnspan=3, pady=2)

        # 摄像头控制按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=10)

        self.camera_btn = ttk.Button(btn_frame, text="开启摄像头", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=3)

        self.capture_btn = ttk.Button(btn_frame, text="截图比对", command=self.capture_and_compare, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=3)

        # 摄像头状态
        self.camera_status = ttk.Label(left_frame, text="摄像头状态: 已关闭", foreground='gray')
        self.camera_status.grid(row=7, column=0, columnspan=3, pady=2)

        # 比对结果
        self.camera_result = ttk.Label(left_frame, text="", font=("Microsoft YaHei", 9), wraplength=200, justify='left')
        self.camera_result.grid(row=8, column=0, columnspan=3, pady=2)

        # 右侧：视频显示区域
        right_frame = ttk.LabelFrame(content_frame, text="摄像头预览", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=5)

        # 初始化空白画面
        blank_img = np.zeros((480, 480, 3), dtype=np.uint8)
        blank_pil = Image.fromarray(blank_img)
        self.blank_photo = ImageTk.PhotoImage(blank_pil)

        self.video_label = tk.Label(right_frame, image=self.blank_photo, relief=tk.SUNKEN, anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    # ==================== 功能方法 ====================

    def select_img1(self):
        path = filedialog.askopenfilename(title="选择第一张人脸图片",
                                          filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.img1_path = path
            self.img1_label.config(text=os.path.basename(path))
            self.show_image_preview(path, self.img1_preview)

    def select_img2(self):
        path = filedialog.askopenfilename(title="选择第二张人脸图片",
                                          filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.img2_path = path
            self.img2_label.config(text=os.path.basename(path))
            self.show_image_preview(path, self.img2_preview)

    def select_query(self):
        path = filedialog.askopenfilename(title="选择查询人脸图片",
                                          filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.query_path = path
            self.query_label.config(text=os.path.basename(path))
            self.show_image_preview(path, self.query_preview)

    def select_gallery(self):
        path = filedialog.askopenfilename(title="选择人群照片",
                                          filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.gallery_path = path
            self.gallery_label.config(text=os.path.basename(path))
            self.show_image_preview(path, self.gallery_preview)

    def select_ref_face(self):
        path = filedialog.askopenfilename(title="选择基准人脸图片",
                                          filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.ref_face_path = path
            self.ref_label.config(text=os.path.basename(path))
            # 预览尺寸缩小到一半
            self.show_image_preview(path, self.ref_preview, max_size=(200, 200))

            # 提取基准人脸特征
            try:
                img = cv2.imread(path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.app.get(img_rgb)
                if len(faces) > 0:
                    self.ref_feat = faces[0].embedding
                    self.ref_status.config(text="✓ 基准人脸特征已提取", foreground='green')
                else:
                    self.ref_feat = None
                    self.ref_status.config(text="✗ 未检测到人脸", foreground='red')
            except Exception as e:
                self.ref_feat = None
                self.ref_status.config(text=f"✗ 特征提取失败: {e}", foreground='red')

    def show_image_preview(self, path, label_widget, max_size=(300, 300)):
        """显示图片预览"""
        try:
            img = Image.open(path)
            # 先强制缩放到 max_size，保持宽高比
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            label_widget.config(text=f"预览失败: {e}")

    def draw_faces_on_image(self, img, faces, highlight_idx=-1):
        """在人脸图像上绘制人脸框"""
        img_copy = img.copy()
        for i, face in enumerate(faces):
            bbox = face.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # 判断是否是最相似人脸
            if i == highlight_idx:
                color = (0, 255, 0)  # 绿色
                thickness = 3
            else:
                color = (255, 0, 0)  # 蓝色
                thickness = 2

            # 绘制人脸框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

            # 添加序号标签
            label = f"#{i+1}"
            cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img_copy

    def compare_two(self):
        """比对两张人脸"""
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("提示", "请先选择两张人脸图片")
            return

        try:
            self.result_label1.config(text="正在比对，请稍候...")
            self.root.update()

            # 用 cv2 读取图片并转为 RGB 格式
            img1 = cv2.imread(self.img1_path)
            img2 = cv2.imread(self.img2_path)

            if img1 is None or img2 is None:
                self.result_label1.config(text="✗ 错误: 无法读取图片")
                return

            # insightface 需要 RGB 格式
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            faces1 = self.app.get(img1_rgb)
            faces2 = self.app.get(img2_rgb)

            if len(faces1) == 0 or len(faces2) == 0:
                self.result_label1.config(text="✗ 错误: 未检测到人脸")
                return

            feat1 = faces1[0].embedding
            feat2 = faces2[0].embedding

            sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

            result_text = f"\n相似度: {sim:.4f}\n\n"
            if sim > 0.5:
                result_text += "✓ 判断: 同一人 (可能性较高)"
            elif sim > 0.35:
                result_text += "○ 判断: 可能是同一人"
            else:
                result_text += "✗ 判断: 不同人"

            self.result_label1.config(text=result_text)

        except Exception as e:
            messagebox.showerror("错误", f"比对失败: {e}")

    def search_face(self):
        """在人群中搜索人脸并在图上标注"""
        if not self.query_path or not self.gallery_path:
            messagebox.showwarning("提示", "请先选择查询人脸和人群照片")
            return

        try:
            self.result_label2.config(text="正在搜索，请稍候...")
            self.root.update()

            # 用 cv2 读取图片并转为 RGB 格式
            query_img = cv2.imread(self.query_path)
            gallery_img = cv2.imread(self.gallery_path)

            if query_img is None or gallery_img is None:
                self.result_label2.config(text="✗ 错误: 无法读取图片")
                return

            query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            gallery_img_rgb = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

            query_faces = self.app.get(query_img_rgb)
            gallery_faces = self.app.get(gallery_img_rgb)

            if len(query_faces) == 0:
                self.result_label2.config(text="✗ 未在查询图片中检测到人脸")
                return

            if len(gallery_faces) == 0:
                self.result_label2.config(text="✗ 未在人群照片中检测到人脸")
                return

            query_feat = query_faces[0].embedding

            similarities = []
            for i, face in enumerate(gallery_faces):
                sim = np.dot(query_feat, face.embedding) / (np.linalg.norm(query_feat) * np.linalg.norm(face.embedding))
                similarities.append((i, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            # 获取最相似人脸的索引
            best_idx = similarities[0][0]
            best_sim = similarities[0][1]

            # 在人群照片上绘制所有检测到的人脸框，最相似的用绿色标注
            result_img = self.draw_faces_on_image(gallery_img, gallery_faces, highlight_idx=best_idx)

            # 转换显示格式
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            result_pil.thumbnail((500, 500), Image.Resampling.LANCZOS)
            result_photo = ImageTk.PhotoImage(result_pil)
            self.gallery_preview.config(image=result_photo, text="")
            self.gallery_preview.image = result_photo

            result_text = f"检测到 {len(gallery_faces)} 张人脸\n\n"
            result_text += f"最相似人脸: 人脸 {best_idx+1}\n"
            result_text += f"相似度: {best_sim:.4f}\n\n"
            if best_sim > 0.5:
                result_text += "✓ 判断: 人群中包含同一人"
            elif best_sim > 0.35:
                result_text += "○ 判断: 可能是同一人"
            else:
                result_text += "✗ 判断: 人群中不含此人"

            self.result_label2.config(text=result_text)

        except Exception as e:
            messagebox.showerror("错误", f"搜索失败: {e}")

    def toggle_camera(self):
        """开关摄像头"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.camera_running = True
        self.camera_btn.config(text="关闭摄像头")
        self.camera_status.config(text="摄像头状态: 运行中", foreground='green')
        self.capture_btn.config(state=tk.NORMAL)

        # 如果有基准人脸特征，开始实时人脸检测比对
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        """停止摄像头"""
        self.camera_running = False  # 先停止循环
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="开启摄像头")
        self.camera_status.config(text="摄像头状态: 已关闭", foreground='gray')
        self.capture_btn.config(state=tk.DISABLED)
        # 等待一小段时间确保线程停止后再切换画面
        self.root.after(100, self._show_blank_screen)

    def _show_blank_screen(self):
        """在主线程中显示空白画面"""
        self.video_label.config(image=self.blank_photo)

    def camera_loop(self):
        """摄像头循环，支持实时人脸检测"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 水平翻转
            frame = cv2.flip(frame, 1)

            # 如果有基准人脸，进行人脸检测和比对
            if self.ref_feat is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(frame_rgb)

                for face in faces:
                    # 计算与基准人脸的相似度
                    sim = np.dot(self.ref_feat, face.embedding) / (np.linalg.norm(self.ref_feat) * np.linalg.norm(face.embedding))

                    bbox = face.bbox
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # 根据相似度设置颜色
                    if sim > 0.5:
                        color = (0, 255, 0)  # 绿色 - 匹配
                        label = f"MATCH: {sim:.2f}"
                    elif sim > 0.35:
                        color = (0, 255, 255)  # 黄色 - 可能
                        label = f"MAYBE: {sim:.2f}"
                    else:
                        color = (0, 0, 255)  # 红色 - 不匹配
                        label = f"DIFF: {sim:.2f}"

                    # 绘制人脸框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # 绘制标签
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # 无基准人脸时只显示普通检测结果
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(frame_rgb)
                for face in faces:
                    bbox = face.bbox
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo

    def capture_and_compare(self):
        """截图并与基准人脸比对"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("提示", "请先开启摄像头")
            return

        if self.ref_feat is None:
            messagebox.showwarning("提示", "请先上传基准人脸")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("错误", "截图失败")
            return

        # 水平翻转
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测人脸
        faces = self.app.get(frame_rgb)

        if len(faces) == 0:
            self.camera_result.config(text="✗ 未检测到人脸", foreground='red')
            return

        # 查找最相似的人脸
        best_sim = 0
        best_idx = 0
        for i, face in enumerate(faces):
            sim = np.dot(self.ref_feat, face.embedding) / (np.linalg.norm(self.ref_feat) * np.linalg.norm(face.embedding))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # 更新结果显示
        result_text = f"检测到 {len(faces)} 张人脸\n"
        result_text += f"最相似: {best_sim:.4f}\n"
        if best_sim > 0.5:
            result_text += "✓ 匹配成功: 同一人"
        elif best_sim > 0.35:
            result_text += "○ 可能匹配"
        else:
            result_text += "✗ 不匹配"

        self.camera_result.config(text=result_text, foreground='green' if best_sim > 0.5 else 'orange')

        # 在帧上绘制结果并显示到左侧截图结果区域
        result_frame = self.draw_faces_on_image(frame, faces, highlight_idx=best_idx)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        result_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
        result_photo = ImageTk.PhotoImage(result_pil)
        self.capture_result_label.config(image=result_photo, text="")
        self.capture_result_label.image = result_photo

        # 保存当前帧用于后续保存
        self.current_frame = frame

    def save_screenshot(self):
        """保存当前截图"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            filename = f"capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("提示", f"截图已保存: {filename}")
        else:
            messagebox.showwarning("提示", "请先点击截图比对")

    def on_closing(self):
        """关闭程序"""
        self.stop_camera()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()