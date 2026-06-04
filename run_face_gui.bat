@echo off
chcp 65001 >nul
echo ============================================
echo 人脸识别系统 - 启动脚本
echo ============================================
echo.
echo 正在使用 anaconda 环境: gluoncv-new
echo.
cd /d "%~dp0"
D:\leng\AI\anaconda3\envs\gluoncv-new\python.exe face_gui.py
pause