@echo off
chcp 65001 >nul
echo ============================================
echo  Safety-Helmet-Wearing-Dataset
echo  一键安装人脸识别环境
echo  Target Env: gluoncv-new
echo ============================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 激活 anaconda 环境
call D:\leng\AI\anaconda3\Scripts\activate.bat gluoncv-new
if errorlevel 1 (
    echo [ERROR] 激活环境失败，请检查 anaconda 路径
    pause
    exit /b 1
)

echo [1/3] 升级 pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [2/3] 安装人脸识别核心依赖...
python -m pip install -r requirements_face.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [3/3] 验证安装...
python -c "import insightface; import onnxruntime; import cv2; import numpy; from PIL import Image; print('insightface:', insightface.__version__); print('onnxruntime:', onnxruntime.__version__); print('opencv:', cv2.__version__); print('numpy:', numpy.__version__); print('Pillow OK')"

echo.
echo ============================================
echo  安装完成！可运行 run_face_gui.bat 启动
echo ============================================
pause
