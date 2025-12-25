#!/bin/bash
# YOLOv5 Training Optimization - 环境设置脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "YOLOv5 Training Optimization - 环境设置"
echo "========================================"
echo ""

# 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo ""
    echo "2. 创建虚拟环境..."
    python -m venv venv
    echo "   ✓ 虚拟环境已创建: venv/"

    # 激活虚拟环境
    echo ""
    echo "3. 激活虚拟环境..."
    source venv/bin/activate
    echo "   ✓ 虚拟环境已激活"
fi

# 检查CUDA
echo ""
echo "4. 检查CUDA环境..."
python -c "import torch; print(f'   PyTorch版本: {torch.__version__}'); print(f'   CUDA可用: {torch.cuda.is_available()}'); print(f'   CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'   GPU数量: {torch.cuda.device_count()}'); print(f'   GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}') 2>/dev/null || echo "   ⚠ PyTorch未安装"

# 安装依赖
echo ""
read -p "是否安装依赖包? (y/n): " install_deps
if [ "$install_deps" = "y" ]; then
    echo ""
    echo "5. 安装依赖包..."
    echo "   安装基础依赖..."
    pip install -r requirements.txt

    echo ""
    echo "   检查YOLOv5..."
    if [ ! -d "yolov5" ]; then
        echo "   克隆YOLOv5仓库..."
        git clone https://github.com/ultralytics/yolov5.git
        cd yolov5
        pip install -r requirements.txt
        cd ..
        echo "   ✓ YOLOv5已安装"
    else
        echo "   ✓ YOLOv5目录已存在"
    fi
fi

# 创建数据集目录
echo ""
echo "6. 创建数据集目录结构..."
python scripts/prepare_dataset.py --mode create --output ./data/kitti_subset

# 创建输出目录
echo ""
echo "7. 创建输出目录..."
mkdir -p outputs
echo "   ✓ 输出目录已创建: outputs/"

# 检查GPU
echo ""
echo "8. GPU信息..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "   ⚠ nvidia-smi 未找到"
fi

# 设置脚本执行权限
echo ""
echo "9. 设置脚本权限..."
chmod +x run_all.sh
chmod +x run_baseline.sh
chmod +x scripts/prepare_dataset.py 2>/dev/null || true
chmod +x scripts/compare_experiments.py 2>/dev/null || true
echo "   ✓ 脚本权限已设置"

# 完成
echo ""
echo "========================================"
echo "环境设置完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 准备数据集: 将数据放入 data/kitti_subset/ 目录"
echo "  2. 运行实验: python train.py --experiment baseline"
echo "  3. 或运行所有实验: ./run_all.sh"
echo ""
echo "文档: 查看 README.md 获取详细说明"
echo ""

# 如果创建了虚拟环境，提示如何激活
if [ "$create_venv" = "y" ]; then
    echo "虚拟环境激活命令:"
    echo "  source venv/bin/activate"
    echo ""
fi
