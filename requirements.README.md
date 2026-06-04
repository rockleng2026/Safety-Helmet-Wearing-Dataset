# 依赖文件说明

本目录提供两个 requirements 文件，分别面向不同模块：

## 文件清单

| 文件 | 适用模块 | 说明 |
|------|----------|------|
| `requirements_face.txt` | `face_gui.py`、`face_compare_*.py` | 人脸识别相关**最小依赖** |
| `requirements.txt` | 整个工程 | 包含 GluonCV、PyQt5 等**完整依赖** |

## 推荐安装方式（人脸识别 GUI）

`gluoncv-new` 环境基于 Python 3.7 + MXNet + GluonCV，**已包含**大量依赖，仅需补装人脸识别部分：

```bash
# 激活环境
conda activate gluoncv-new

# 进入项目目录
cd D:\leng\AI\Safety-Helmet-Wearing-Dataset

# 安装人脸识别最小依赖
pip install -r requirements_face.txt
```

或直接双击 `install_face_env.bat` 一键安装。

## 核心依赖版本说明

`gluoncv-new` 是 Python 3.7 环境，因此版本做了兼容：

| 库 | 版本约束 | 原因 |
|----|----------|------|
| `insightface` | `>=0.7.3` | 兼容 Python 3.7 |
| `onnxruntime` | `>=1.14.1` | 兼容 Python 3.7 |
| `numpy` | `<1.24.0` | mxnet 1.8 不支持 numpy 1.24+ |
| `protobuf` | `>=3.20.0` | 兼容 onnx 系列 |

## 验证安装

```bash
python -c "from insightface.app import FaceAnalysis; print('OK')"
```

成功输出 `OK` 即表示环境就绪。

## 启动

```bash
# 方式 1: 命令行
python face_gui.py

# 方式 2: 启动脚本（双击）
run_face_gui.bat
```

## YOLOv8 注意事项

`gluoncv-new` (Python 3.7) **无法**运行 YOLOv8 相关脚本，需新建 Python 3.10+ 环境：

```bash
conda create -n yolov8-shwd python=3.10 -y
conda activate yolov8-shwd
pip install ultralytics torch torchvision PyQt5
```
