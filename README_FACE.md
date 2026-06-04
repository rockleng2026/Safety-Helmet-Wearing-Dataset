# 人脸识别系统 - 环境部署与使用说明

---

## 1. 系统概述

人脸识别系统包含三个功能模块：

| 模块 | 功能 | 使用方式 |
|------|------|----------|
| `face_compare_two.py` | 比对2张人脸图片的相似度 | 命令行 |
| `face_compare_one_to_many.py` | 在人群照片中搜索匹配的人脸 | 命令行 |
| `face_gui.py` | GUI图形界面，包含实时摄像头人脸识别 | 图形界面 |

---

## 2. 环境要求

### 2.1 硬件要求

- CPU 或 NVIDIA GPU（推荐）
- 摄像头（用于实时识别）

### 2.2 软件要求

- **Anaconda 3**：已配置 `gluoncv-new` 环境
- **Python 3.12**：已安装在 Anaconda 环境中
- **必需依赖**：
  - `insightface`：人脸识别核心库
  - `opencv-python`：图像处理
  - `numpy`：数值计算
  - `Pillow`：图像显示
  - `tkinter`：GUI界面（Python内置）

---

## 3. 环境部署

### 3.1 激活Anaconda环境

```bash
conda activate D:\leng\AI\anaconda3\envs\gluoncv-new
```

### 3.2 安装依赖（如未安装）

```bash
pip install insightface opencv-python numpy Pillow
```

### 3.3 验证环境

```bash
python -c "from insightface.app import FaceAnalysis; print('insightface OK')"
```

---

## 4. 使用方法

### 4.1 命令行模式

#### 4.1.1 两张人脸比对

```bash
python face_compare_two.py --img1 图片1路径 --img2 图片2路径
```

**示例**：
```bash
python face_compare_two.py --img1 ./image/face/1.jpg --img2 ./image/face/2.jpg
```

**输出说明**：
```
相似度: 0.8523

✓ 判断: 同一人 (可能性较高)
```

相似度阈值：
- `> 0.5`：同一人
- `0.35 ~ 0.5`：可能是同一人
- `< 0.35`：不同人

---

#### 4.1.2 人脸搜索（人群中查找）

```bash
python face_compare_one_to_many.py --query 查询人脸路径 --gallery 人群照片路径
```

**示例**：
```bash
python face_compare_one_to_many.py --query ./image/face/1.jpg --gallery ./image/group.jpg
```

**输出说明**：
```
检测到 5 张人脸

  人脸 1: 相似度=0.8523, 检测置信度=0.95
  人脸 2: 相似度=0.1234, 检测置信度=0.92
  ...

最相似人脸: 人脸 1
相似度: 0.8523

✓ 判断: 人群中包含同一人
```

---

### 4.2 GUI模式（推荐）

#### 4.2.1 启动方式

**方式一**：双击运行启动脚本
```
双击文件: run_face_gui.bat
```

**方式二**：命令行运行
```bash
python face_gui.py
```

> 注意：必须使用 Anaconda 环境中的 Python 运行

#### 4.2.2 功能说明

##### Tab 1：两张人脸比对

1. 点击「选择图片1」上传第一张人脸图片
2. 点击「选择图片2」上传第二张人脸图片
3. 点击「开始比对」查看相似度结果

##### Tab 2：人脸搜索

1. 点击「选择查询图片」上传单人脸图片
2. 点击「选择人群照片」上传包含多人的图片
3. 点击「开始搜索并标注」查看结果

**功能亮点**：
- 人群照片上会标注所有检测到的人脸
- 最相似的人脸用**绿色框**高亮显示
- 其他人员用蓝色框标注

##### Tab 3：摄像头人脸识别

**布局**：
- 左侧：基准人脸上传 + 截图结果显示 + 控制按钮
- 右侧：摄像头实时预览

**操作步骤**：
1. 点击「上传」选择基准人脸图片（需单人正面照片）
2. 等待特征提取成功后（显示绿色提示），点击「开启摄像头」
3. 摄像头开启后，实时检测人脸并显示相似度标签
4. 点击「截图比对」可截图当前画面进行比对

**相似度颜色编码**：
| 颜色 | 相似度 | 判断 |
|------|--------|------|
| 绿色 | > 0.5 | 匹配成功 |
| 黄色 | 0.35~0.5 | 可能匹配 |
| 红色 | < 0.35 | 不匹配 |

---

## 5. 常见问题

### Q1: 提示 "ModuleNotFoundError: No module named 'insightface'"

**原因**：未使用 Anaconda 环境运行

**解决**：
```bash
conda activate D:\leng\AI\anaconda3\envs\gluoncv-new
D:\leng\AI\anaconda3\envs\gluoncv-new\python.exe face_gui.py
```

或双击 `run_face_gui.bat` 运行

---

### Q2: 上传基准人脸后提示 "特征提取失败"

**原因**：
- 人脸图片不清晰
- 图片中有多个人脸
- 光线不足或角度不佳

**解决**：上传单人正面、清晰、光线充足的人脸图片

---

### Q3: 摄像头无法打开

**原因**：摄像头被其他程序占用

**解决**：关闭其他使用摄像头的程序后重试

---

### Q4: 比对结果不准确

**原因**：人脸特征受表情、角度、光线影响

**建议**：
- 使用正面清晰的照片作为基准人脸
- 确保比对时环境光线一致
- 相似度在阈值附近时可多次比对验证

---

## 6. 技术参数

| 参数 | 值 |
|------|-----|
| 人脸检测模型 | buffalo_l（InsightFace） |
| 特征维度 | 512维 |
| 相似度算法 | 余弦相似度 |
| 匹配阈值 | 0.5（默认） |

---

## 7. 文件说明

```
D:\leng\AI\Safety-Helmet-Wearing-Dataset\
├── face_compare_two.py         # 两张人脸比对脚本
├── face_compare_one_to_many.py # 人脸搜索脚本
├── face_gui.py                # GUI图形界面
├── run_face_gui.bat           # 启动脚本（自动使用Anaconda环境）
├── image/face/                 # 测试人脸图片
└── README_FACE.md             # 本文档
```

---

*文档版本：V1.0*
*生成日期：2026年6月*