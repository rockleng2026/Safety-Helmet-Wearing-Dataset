# -*- coding: utf-8 -*-
"""
Safety Helmet Wearing Detection UI - YOLOv8 Version
基于 Ultralytics YOLOv8 + PyQt5 界面
参考 yolov5-master/main.py 布局
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.is_camera = False

        # YOLOv8 模型参数
        self.model = None
        self.model_path = 'runs/detect/shwd/weights/best.pt'
        self.imgsz = 416
        self.conf_threshold = 0.4
        self.device = '0'  # GPU

        # 类别和颜色 (与原 GluonCV 版本一致: hat=红, person=蓝)
        self.classes = ['hat', 'person']
        self.colors = [(0, 0, 255), (255, 0, 0)]  # hat=红, person=蓝 (BGR)

        self.init_model()

    def init_model(self):
        """初始化 YOLOv8 模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f'YOLOv8 model loaded: {self.model_path}')
            else:
                print(f'Warning: Model not found at {self.model_path}')
            self.model_loaded = True
        except Exception as e:
            print(f'Model init error: {e}')
            self.model_loaded = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 主垂直布局
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.mainLayout.setObjectName("mainLayout")
        self.mainLayout.setContentsMargins(5, 5, 5, 5)

        # 上部水平布局（按钮 + 显示区）
        self.topLayout = QtWidgets.QHBoxLayout()
        self.topLayout.setObjectName("topLayout")

        # 左侧按钮垂直布局
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName("verticalLayout")

        # 图片检测按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(140, 80))
        self.pushButton_img.setMaximumSize(QtCore.QSize(140, 80))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(11)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)

        # 摄像头检测按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(140, 80))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(140, 80))
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)

        # 视频检测按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(140, 80))
        self.pushButton_video.setMaximumSize(QtCore.QSize(140, 80))
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)

        # 添加弹簧使按钮靠上
        self.verticalLayout.addStretch()

        # 右侧显示区域
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.label.setMinimumSize(QtCore.QSize(800, 550))

        # 组装上部布局
        self.topLayout.addLayout(self.verticalLayout)
        self.topLayout.addWidget(self.label, 1)

        # 添加到主布局
        self.mainLayout.addLayout(self.topLayout)

        # 底部状态标签
        self.status_label = QtWidgets.QLabel(self.centralwidget)
        self.status_label.setObjectName("status_label")
        self.status_label.setFont(QtGui.QFont("Microsoft YaHei", 9))
        self.status_label.setStyleSheet("color: #888; padding: 3px; background-color: #f0f0f0;")
        self.status_label.setText("状态: 就绪 | 模型: YOLOv8 | 类别: hat(安全帽), person(人像)")
        self.mainLayout.addWidget(self.status_label)

        MainWindow.setCentralWidget(self.centralwidget)

        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "安全帽佩戴检测系统 - YOLOv8"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", ""))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_camera_frame)

    def detect_and_draw(self, orig_img):
        """使用 YOLOv8 检测并绘制边界框"""
        if self.model is None:
            return orig_img

        try:
            # YOLOv8 推理
            results = self.model(orig_img, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

            # 绘制边界框（与原 GluonCV 版本一致: hat=红, person=蓝）
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = self.colors[cls_id]
                label = f'{self.classes[cls_id]} {conf:.2f}'

                # 画框
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
                # 画标签背景
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(orig_img, (x1, y1 - t_size[1] - 4),
                             (x1 + t_size[0], y1), color, -1)
                # 画文字
                cv2.putText(orig_img, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return orig_img
        except Exception as e:
            import traceback
            print(f"Detection error: {e}")
            traceback.print_exc()
            return orig_img

    def button_image_open(self):
        """图片检测"""
        self.stop_all()
        print('button_image_open')

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;*.jpeg;;All Files(*)"
        )
        if not img_name:
            return

        img = cv2.imread(img_name)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "错误", "无法读取图片")
            return

        # 检测并绘制
        result_img = self.detect_and_draw(img)

        # 显示到界面
        self.display_image(result_img)
        self.status_label.setText(f"状态: 已检测 | 图片: {os.path.basename(img_name)}")

    def button_video_open(self):
        """视频检测"""
        self.stop_all()
        print('button_video_open')

        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;*.mkv;;All Files(*)"
        )
        if not video_name:
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(self, "警告", "打开视频失败")
            return

        # 获取视频尺寸
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算显示尺寸
        max_width, max_height = 800, 550
        scale = min(max_width / self.video_width, max_height / self.video_height)
        self.display_width = int(self.video_width * scale)
        self.display_height = int(self.video_height * scale)

        # 创建输出 writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('prediction.avi', fourcc, 20, (self.video_width, self.video_height))

        self.timer_video.start(30)
        self.pushButton_video.setDisabled(True)
        self.pushButton_img.setDisabled(True)
        self.pushButton_camera.setDisabled(True)
        self.status_label.setText(f"状态: 视频检测中 | {os.path.basename(video_name)}")
        self.is_camera = False

    def button_camera_open(self):
        """摄像头检测"""
        if not self.timer_video.isActive():
            # 打开摄像头
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, "警告", "打开摄像头失败")
                return

            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 摄像头不使用 VideoWriter
            self.out = None

            self.timer_video.start(100)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setText("关闭摄像头")
            self.status_label.setText("状态: 摄像头检测中 | 按下按钮停止")
            self.is_camera = True
        else:
            self.stop_all()
            self.reset_buttons()
            self.status_label.setText("状态: 已停止")

    def show_camera_frame(self):
        """显示视频/摄像头帧"""
        flag, img = self.cap.read()
        if img is None:
            self.timer_video.stop()
            self.cap.release()
            if self.out:
                self.out.release()
            self.reset_buttons()
            self.status_label.setText("状态: 播放结束")
            return

        # 检测并绘制
        result_img = self.detect_and_draw(img)

        # 只在视频检测时写入文件
        if self.out and self.out.isOpened():
            self.out.write(result_img)

        # 显示到界面
        if self.is_camera:
            self.display_image(result_img, width=640, height=480)
        else:
            self.display_image(result_img, width=self.display_width, height=self.display_height)

    def display_image(self, img, width=800, height=600):
        """将 OpenCV 图片显示到 QLabel"""
        show = cv2.resize(img, (width, height))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        h, w, ch = show.shape
        bytes_per_line = 3 * w
        qt_image = QtGui.QImage(show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        self.label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def stop_all(self):
        """停止所有视频流"""
        self.timer_video.stop()
        if self.cap.isOpened():
            self.cap.release()
        if self.out:
            self.out.release()
            self.out = None
        self.is_camera = False

    def reset_buttons(self):
        """重置按钮状态"""
        self.pushButton_video.setDisabled(False)
        self.pushButton_img.setDisabled(False)
        self.pushButton_camera.setDisabled(False)
        self.pushButton_camera.setText("摄像头检测")

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        self.stop_all()
        event.accept()


if __name__ == '__main__':
    # Windows 路径兼容
    if sys.platform.startswith('win'):
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())