# 安全帽佩戴检测系统 - 环境搭建与使用指南

## 📋 目录
- [环境要求](#环境要求)
- [环境搭建](#环境搭建)
- [模型下载](#模型下载)
- [快速使用](#快速使用)
- [GUI 图形界面](#gui-图形界面)
- [命令行工具](#命令行工具)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.7.x |
| Anaconda | 任意版本 |
| MXNet | ≥ 1.8.0 |
| GluonCV | ≥ 0.10.0 |
| OpenCV | ≥ 4.0 |

---

## 环境搭建

### 1. 创建 Conda 环境

```bash
# 使用 Python 3.7 创建新环境
conda create -n gluoncv-new python=3.7 -y

# 激活环境
conda activate gluoncv-new
```

### 2. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 3. 安装 MXNet 和 GluonCV

```bash
pip install --upgrade mxnet gluoncv --upgrade-strategy eager
```

### 4. 安装 OpenCV

```bash
pip install opencv-python
```

### 5. 安装 PyQt5（用于 GUI）

```bash
pip install PyQt5==5.15.6
```

---

## 模型下载

### 下载地址

| 模型 | 百度网盘 | Google Drive |
|------|----------|--------------|
| darknet53 | [下载](https://pan.baidu.com/s/1dWNU_q59sw1a3TVtV7VXEg) | [下载](https://drive.google.com/open?id=1_0A-bQbsprzStefQOMiQpLn8JZBchqho) |
| mobilenet1.0 | 同上 | 同上 |
| mobile0.25 | 同上 | 同上 |

### 模型文件说明

| 文件名 | 网络结构 | 推荐场景 |
|--------|----------|----------|
| `darknet.params` | darknet53 | 精度最高，速度最慢 |
| `mobilenet1.0.params` | mobilenet1.0 | 精度中等，速度中等 ⭐ |
| `mobile0.25.params` | mobilenet0.25 | 精度较低，速度最快 |

### 放置位置

下载后将模型文件放入 `models/` 目录：

```
Safety-Helmet-Wearing-Dataset/
└── models/
    ├── darknet.params      # 可选
    ├── mobilenet1.0.params # 推荐使用
    └── mobile0.25.params   # 可选
```

---

## 快速使用

### 图片检测

```bash
# 进入项目目录
cd D:\leng\AI\Safety-Helmet-Wearing-Dataset

# 使用 mobilenet1.0 模型检测图片（CPU 模式）
python test_yolo.py --network yolo3_mobilenet1.0_voc --image 11.jpg --cpu

# 使用 darknet 模型（更高精度）
python test_yolo.py --network yolo3_darknet53_voc --image 11.jpg --cpu

# 使用 GPU 加速（如已安装 CUDA）
python test_yolo.py --network yolo3_mobilenet1.0_voc --image 11.jpg
```

### 视频检测

```bash
python test_yolo_video.py --video videos/3.mp4 --output 3_detected.mp4 --cpu
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--network` | 模型网络结构 | yolo3_darknet53_voc |
| `--model-path` | 模型文件目录 | models |
| `--short` | 输入尺寸（320/416/512/608） | 416 |
| `--threshold` | 检测置信度阈值 | 0.4 |
| `--image` | 输入图片路径 | - |
| `--video` | 输入视频路径 | - |
| `--cpu` | 使用 CPU 而不是 GPU | - |

---

## GUI 图形界面

### 启动 GUI

```bash
python gui_app.py
```

### 功能说明

| 按钮 | 功能 |
|------|------|
| 图片检测 | 选择图片文件进行检测，显示带标注的结果 |
| 视频检测 | 选择视频文件，实时显示检测结果 |
| 摄像头检测 | 打开本地摄像头，实时检测安全帽佩戴情况 |

### 状态栏说明

底部状态栏显示当前状态、使用的模型和类别信息。

---

## 命令行工具

### 训练模型

```bash
python train_yolo.py --batch-size 4 -j 4 --warmup-epochs 3 --epochs 100
```

训练参数配置在 `train_yolo.py` 的 `get_dataset()` 函数中修改路径。

### 符号模型推理

```bash
python test_symbol.py
```

---

## 项目结构

```
Safety-Helmet-Wearing-Dataset/
├── models/                  # 模型文件目录
│   ├── darknet.params
│   ├── mobilenet1.0.params
│   └── mobile0.25.params
├── image/                   # 示例图片
├── videos/                  # 测试视频
├── symbol/                  # 符号模型文件
├── test_yolo.py             # 图片检测脚本
├── test_yolo_video.py       # 视频检测脚本
├── test_symbol.py           # 符号模型推理
├── train_yolo.py            # 训练脚本
├── gui_app.py               # GUI 图形界面 ⭐
└── README.md                # 本文档
```

---

## 常见问题

### Q: 提示 "No module named 'gluoncv'"？

**解决**：确保已激活 `gluoncv-new` 环境，并已安装 gluoncv：
```bash
conda activate gluoncv-new
pip install mxnet gluoncv
```

### Q: 提示 "No module named 'PyQt5'"？

**解决**：
```bash
pip install PyQt5==5.15.6
```

### Q: GPU 模式报错？

**解决**：确认 MXNet 已正确安装 GPU 版本。如需使用 GPU，需安装 CUDA 和 cuDNN，然后：
```bash
pip install mxnet-cu101  # CUDA 10.1 版本
```
或根据你的 CUDA 版本选择对应版本。

### Q: 图片检测显示 "Invalid argument"？

**解决**：确保图片路径使用绝对路径或相对路径正确，避免中文路径。

### Q: 视频/摄像头卡顿？

**解决**：这是正常现象，GluonCV YOLO 推理比 PyTorch YOLOv5 慢。可以：
1. 使用更小的 `--short` 参数（如 320）
2. 使用更轻量的 `mobile0.25` 模型
3. 降低 GUI 帧率

### Q: 模型文件太大无法 push？

**解决**：GitHub 建议使用 Git LFS 管理大文件。文件 > 50MB 会有警告但仍可上传。

---

## 📚 相关链接

- 数据集下载：[BaiduDrive](https://pan.baidu.com/s/1UbFkGm4EppdAU660Vu7SdQ) | [GoogleDrive](https://drive.google.com/open?id=1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX)
- 模型下载：[BaiduDrive](https://pan.baidu.com/s/1dWNU_q59sw1a3TVtV7VXEg) | [GoogleDrive](https://drive.google.com/open?id=1_0A-bQbsprzStefQOMiQpLn8JZBchqho)
- 原始项目：[github.com/njvisionpower/Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/SafetyHelmetWearing-Dataset)

---

## 🚀 YOLOv8 迁移指南 (性能升级)

GluonCV YOLO3 推理速度较慢（3-8 FPS），迁移到 YOLOv8 可提升 **5-10 倍** 性能。

### 适用场景

- 视频/摄像头实时检测（目标 25+ FPS）
- 训练时间充足（有 NVIDIA GPU）
- 需要更高精度

### 迁移文件清单

| 文件 | 用途 | 执行位置 |
|------|------|----------|
| `convert_voc_to_yolo.py` | VOC 数据集 → YOLO 格式转换 | 笔记本 |
| `train_yolov8.py` | YOLOv8 训练脚本 | 台式机 (3060) |
| `test_yolov8.py` | YOLOv8 图片检测 | 笔记本 |
| `test_yolov8_video.py` | YOLOv8 视频检测 | 笔记本 |
| `gui_app_yolov8.py` | YOLOv8 GUI 图形界面 | 笔记本 |
| `data.yaml` | Ultralytics 数据集配置 | 笔记本 → 台式机 |

### 迁移步骤

#### 步骤 1: 笔记本 - 数据集格式转换

```bash
# 确认 VOC 数据集路径
# 修改 convert_voc_to_yolo.py 中的 VOC_ROOT 为实际路径（如 D:\VOCdevkit\VOC2028）

# 运行转换
python convert_voc_to_yolo.py --voc-root D:\VOCdevkit\VOC2028 --output dataset

# 生成 dataset/ 和 data.yaml
```

#### 步骤 2: 传输文件到 3060 台式机

将以下文件复制到台式机：
- `dataset/` 目录
- `data.yaml`
- `train_yolov8.py`

#### 步骤 3: 台式机 - 环境搭建

```bash
conda create -n yolov8-shwd python=3.10 -y
conda activate yolov8-shwd

# 安装 PyTorch CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics
pip install ultralytics opencv-python PyQt5 PyYAML tqdm

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### 步骤 4: 台式机 - 开始训练

```bash
python train_yolov8.py --data data.yaml --model yolov8s.pt --epochs 100 --batch 32 --device 0
```

**训练预估时间**: RTX 3060 上 100 epochs ≈ 2-4 小时

#### 步骤 5: 传输模型回笔记本

训练完成后，将 `runs/detect/shwd/weights/best.pt` 复制回笔记本。

#### 步骤 6: 笔记本 - 使用 YOLOv8 推理

```bash
# 图片检测
python test_yolov8.py --model best.pt --image 11.jpg

# 视频检测
python test_yolov8_video.py --model best.pt --video videos/3.mp4 --output result.avi

# GUI 界面
python gui_app_yolov8.py
```

### 性能对比

| 场景 | GluonCV (mobilenet1.0) | YOLOv8 (s) |
|------|------------------------|------------|
| 图片检测 | 0.5-1s/张 | 0.05-0.1s/张 |
| 视频 (1080p) | 3-5 FPS | 25-40 FPS |
| 摄像头实时 | 2-4 FPS | 30+ FPS |

### 环境要求对比

| 组件 | GluonCV | YOLOv8 |
|------|---------|--------|
| Python | 3.7 | 3.10+ |
| GPU | 可选 | 推荐 NVIDIA |
| CUDA | 10.1+ | 11.8+ |
| 训练时间 | 很长 | 较短 |

---

*最后更新：2026-06-02*