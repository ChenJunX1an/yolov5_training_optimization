# 快速开始指南

## 5分钟上手 YOLOv5 训练优化实验

### 第一步：环境设置

```bash
# 克隆项目后，运行设置脚本
bash setup.sh          # Linux/Mac
# 或
setup.bat              # Windows
```

### 第二步：准备数据

```bash
# 创建数据集目录
python scripts/prepare_dataset.py --mode create --output ./data/kitti_subset

# 将数据放入相应目录
# - 训练图像 -> data/kitti_subset/train/images/
# - 训练标签 -> data/kitti_subset/train/labels/
# - 验证图像 -> data/kitti_subset/val/images/
# - 验证标签 -> data/kitti_subset/val/labels/
```

### 第三步：运行第一个实验

```bash
# 运行基准组实验（最简单）
python train.py --experiment baseline --model_type s
```

### 第四步：查看结果

训练完成后，查看 `outputs/baseline/` 目录：
- `training.log` - 训练日志
- `baseline_results.json` - 结果摘要
- `baseline_model.pt` - 训练好的模型

### 第五步：运行所有实验

```bash
./run_all.sh    # Linux/Mac
run_all.bat     # Windows
```

### 第六步：对比结果

```bash
python scripts/compare_experiments.py --output_dir ./outputs
```

查看生成的图表和报告：
- `outputs/comparison_plot.png` - 性能对比图
- `outputs/comparison_report.txt` - 文本报告

## 常用命令

```bash
# 单个实验
python train.py --experiment <experiment_name> --model_type <s|m>

# 自定义参数
python train.py --experiment baseline --epochs 100 --batch_size 16

# 使用YOLOv5m模型
python train.py --experiment baseline --model_type m --batch_size 4

# 验证数据集
python scripts/prepare_dataset.py --mode verify --output ./data/kitti_subset
```

## 实验类型

| 命令 | 说明 |
|------|------|
| `baseline` | 基准组（FP32） |
| `mixed_precision` | 混合精度（FP16） |
| `gradient_checkpoint` | 梯度检查点 |
| `mixed_optimization` | 混合优化 |
| `zero3` | ZeRO-3（DeepSpeed） |
| `ablation` | 消融实验 |

## 故障排除

### 问题：CUDA out of memory
```bash
# 减小batch size
python train.py --experiment baseline --batch_size 4
```

### 问题：找不到YOLOv5
```bash
# 克隆YOLOv5到项目目录
git clone https://github.com/ultralytics/yolov5.git
```

### 问题：依赖安装失败
```bash
# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 更多帮助

查看完整文档：[README.md](README.md)
