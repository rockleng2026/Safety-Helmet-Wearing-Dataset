# -*- coding: utf-8 -*-
"""
Safety Helmet Wearing Detection UI
基于 GluonCV YOLO + PyQt5 界面
参考 yolov5-master/main.py 布局
"""

import sys
import os
import cv2
import numpy as np
import mxnet as mx
from gluoncv import model_zoo, data, utils
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
        self.video_width = 640
        self.video_height = 480
        self.display_width = 640
        self.display_height = 480

        # 初始化 GluonCV YOLO 模型
        self.ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
        self.network = 'yolo3_mobilenet1.0_voc'
        self.model_path = 'models'
        self.short = 416
        self.threshold = 0.4
        self.classes = ['hat', 'person']

        self.init_model()

    def init_model(self):
        """初始化 GluonCV YOLO 模型"""
        try:
            self.net = model_zoo.get_model(self.network, pretrained=False)
            for param in self.net.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
            self.net.reset_class(self.classes)
            self.net.collect_params().reset_ctx(self.ctx)

            # 模型文件映射
            model_files = {
                'yolo3_darknet53_voc': 'darknet.params',
                'yolo3_mobilenet1.0_voc': 'mobilenet1.0.params',
                'yolo3_mobilenet0.25_voc': 'mobile0.25.params'
            }
            model_file = os.path.join(self.model_path, model_files.get(self.network, 'mobilenet1.0.params'))
            if os.path.exists(model_file):
                self.net.load_parameters(model_file, ctx=self.ctx)
                print(f'Model loaded: {model_file}')
            else:
                print(f'Warning: Model file not found: {model_file}, using random init')

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
        self.status_label.setText("状态: 就绪 | 模型: mobilenet1.0 | 类别: hat(安全帽), person(人像)")
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
        MainWindow.setWindowTitle(_translate("MainWindow", "安全帽佩戴检测系统"))
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
        """使用 GluonCV YOLO 检测并绘制边界框（参考 cv_plot_bbox 的 HSV 颜色映射）"""
        try:
            # 转换 BGR (OpenCV) 到 RGB，然后到 MXNet NDArray
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img_nd = mx.nd.array(img_rgb)

            # 使用 GluonCV 的 transform_test
            x, orig = data.transforms.presets.yolo.transform_test([img_nd], short=self.short)
            x = x.as_in_context(self.ctx)

            # 检测
            box_ids, scores, bboxes = self.net(x)

            # 获取检测结果
            ids = box_ids[0].asnumpy()
            scores_np = scores[0].asnumpy()
            bboxes_np = bboxes[0].asnumpy()

            # cv_plot_bbox 颜色逻辑: hsv(cls_id/len(class_names)) -> RGB -> BGR
            # 2 classes: hat=0 -> hsv(0) -> RGB(1,0,0) -> BGR(0,0,255) 红色
            #            person=1 -> hsv(0.5) -> RGB(0,1,1) -> BGR(1,1,0) 青色/蓝绿色
            # 实际 cv_plot_bbox 代码: colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            # hsv(0) matplotlib = 红, hsv(0.5) = 青

            for i in range(len(ids)):
                cls_id = int(ids[i])
                if cls_id < 0:
                    continue
                score = float(scores_np[i])
                if score < self.threshold:
                    continue

                # 边界框坐标
                x_min, y_min, x_max, y_max = bboxes_np[i]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # cv_plot_bbox 原始 BGR 颜色
                # hat: (0, 0, 255) 红色, person: (255, 0, 0) 蓝色
                color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
                label = f'{self.classes[cls_id]} {score:.2f}'

                # 画框
                cv2.rectangle(orig, (x_min, y_min), (x_max, y_max), color, 2)
                # 画标签背景
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(orig, (x_min, y_min - t_size[1] - 4), (x_min + t_size[0], y_min), color, -1)
                # 画文字
                cv2.putText(orig, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 返回 BGR 格式
            return orig[..., ::-1]
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

        # 获取视频原始尺寸
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算显示尺寸（按比例缩放，适应 label 区域）
        max_width = 800
        max_height = 550
        scale = min(max_width / self.video_width, max_height / self.video_height)
        self.display_width = int(self.video_width * scale)
        self.display_height = int(self.video_height * scale)

        # 创建输出 writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('prediction.avi', fourcc, 20, (self.video_width, self.video_height))

        # 开始定时器
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

            # 获取摄像头尺寸
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 摄像头不使用 VideoWriter
            self.out = None

            # 开始定时器 100ms
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

        # 显示到界面（视频按原始比例，摄像头固定）
        if self.is_camera:
            self.display_image(result_img, width=640, height=480)
        else:
            self.display_image(result_img, width=self.display_width, height=self.display_height)

    def display_image(self, img, width=800, height=600):
        """将 OpenCV 图片显示到 QLabel"""
        # 调整大小
        show = cv2.resize(img, (width, height))

        # BGR 转 RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        # 创建 QImage
        h, w, ch = show.shape
        bytes_per_line = 3 * w
        qt_image = QtGui.QImage(show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # 显示
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