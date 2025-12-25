@echo off
REM YOLOv5 Training Optimization - 环境设置脚本 (Windows)

echo ========================================
echo YOLOv5 Training Optimization - 环境设置
echo ========================================
echo.

REM 检查Python版本
echo 1. 检查Python版本...
python --version
if errorlevel 1 (
    echo    错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 创建虚拟环境（可选）
set /p create_venv="是否创建虚拟环境? (y/n): "
if /i "%create_venv%"=="y" (
    echo.
    echo 2. 创建虚拟环境...
    python -m venv venv
    echo    √ 虚拟环境已创建: venv\

    echo.
    echo 3. 激活虚拟环境...
    call venv\Scripts\activate.bat
    echo    √ 虚拟环境已激活
)

REM 检查CUDA
echo.
echo 4. 检查CUDA环境...
python -c "import torch; print(f'   PyTorch版本: {torch.__version__}'); print(f'   CUDA可用: {torch.cuda.is_available()}'); print(f'   CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'   GPU数量: {torch.cuda.device_count()}'); print(f'   GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul || echo    ⚠ PyTorch未安装

REM 安装依赖
echo.
set /p install_deps="是否安装依赖包? (y/n): "
if /i "%install_deps%"=="y" (
    echo.
    echo 5. 安装依赖包...
    echo    安装基础依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo    错误: 依赖安装失败
        pause
        exit /b 1
    )

    echo.
    echo    检查YOLOv5...
    if not exist "yolov5" (
        echo    克隆YOLOv5仓库...
        git clone https://github.com/ultralytics/yolov5.git
        cd yolov5
        pip install -r requirements.txt
        cd ..
        echo    √ YOLOv5已安装
    ) else (
        echo    √ YOLOv5目录已存在
    )
)

REM 创建数据集目录
echo.
echo 6. 创建数据集目录结构...
python scripts\prepare_dataset.py --mode create --output ./data/kitti_subset

REM 创建输出目录
echo.
echo 7. 创建输出目录...
if not exist "outputs" mkdir outputs
echo    √ 输出目录已创建: outputs\

REM 检查GPU
echo.
echo 8. GPU信息...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>nul || echo    ⚠ nvidia-smi 未找到

REM 完成
echo.
echo ========================================
echo 环境设置完成！
echo ========================================
echo.
echo 下一步:
echo   1. 准备数据集: 将数据放入 data\kitti_subset\ 目录
echo   2. 运行实验: python train.py --experiment baseline
echo   3. 或运行所有实验: run_all.bat
echo.
echo 文档: 查看 README.md 获取详细说明
echo.

REM 如果创建了虚拟环境，提示如何激活
if /i "%create_venv%"=="y" (
    echo 虚拟环境激活命令:
    echo   venv\Scripts\activate.bat
    echo.
)

pause
