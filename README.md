# YOLOv5 训练优化实验项目

基于 PyTorch 1.10、CUDA 11.0 环境，适配 RTX 3090 GPU 的 YOLOv5 训练优化方案对比实验。

## 项目概述

本项目实现了6种不同的训练优化方案，用于对比显存占用、训练速度和模型精度（mAP）：

1. **基准组** - 无优化，FP32精度训练
2. **混合精度组** - FP16训练（自动混合精度）
3. **梯度检查点组** - 每4层设置检查点
4. **混合优化组** - FP16 + 每4层检查点
5. **ZeRO-3组** - DeepSpeed优化（支持CPU卸载）
6. **消融实验组** - FP16 + 每2层检查点

## 目录结构

```
yolov5_training_optimization/
├── train.py                     # 主训练脚本
├── requirements.txt             # 依赖清单
├── configs/                     # 配置文件目录
│   ├── baseline_config.json
│   ├── mixed_precision_config.json
│   ├── gradient_checkpoint_config.json
│   ├── mixed_optimization_config.json
│   ├── zero3_config.json
│   └── ablation_config.json
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── data.py                  # 数据加载
│   ├── metrics.py               # 指标统计
│   ├── model.py                 # 模型定义
│   └── loss.py                  # 损失函数
├── scripts/                     # 辅助脚本
│   ├── prepare_dataset.py       # 数据集准备
│   └── compare_experiments.py   # 结果对比
├── run_all.sh/.bat              # 批量运行脚本
└── README.md                    # 项目文档
```

## 环境配置

### 硬件要求

- GPU: NVIDIA RTX 3090 或类似显卡
- 显存: 建议 ≥ 12GB
- 内存: 建议 ≥ 32GB

### 软件依赖

```bash
# 安装 PyTorch (CUDA 11.3)
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安装 DeepSpeed
pip install deepspeed==0.5.10

# 安装 YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5 && pip install -r requirements.txt

# 安装其他依赖
pip install -r requirements.txt
```

或使用 `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 数据集准备

### 1. 创建目录结构

```bash
python scripts/prepare_dataset.py --mode create --output ./data/kitti_subset
```

### 2. 组织数据

将数据按以下结构组织：

```
data/kitti_subset/
├── train/
│   ├── images/          # 训练图像
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/          # YOLO格式标签
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── val/
    ├── images/          # 验证图像
    └── labels/          # 验证标签
```

### 3. 标签格式

标签文件采用 YOLO 格式（归一化坐标）：

```
<class_id> <x_center> <y_center> <width> <height>
```

例如：
```
0 0.5 0.5 0.3 0.4    # 类别0，中心(0.5,0.5)，宽0.3高0.4
1 0.3 0.7 0.2 0.25   # 类别1，中心(0.3,0.7)，宽0.2高0.25
```

### 4. 验证数据集

```bash
python scripts/prepare_dataset.py --mode verify --output ./data/kitti_subset
```

## 使用方法

### 运行单个实验

```bash
# 基准组 (FP32)
python train.py --experiment baseline --model_type s --config configs/baseline_config.json

# 混合精度 (FP16)
python train.py --experiment mixed_precision --model_type s

# 梯度检查点
python train.py --experiment gradient_checkpoint --model_type s

# 混合优化 (FP16 + 检查点)
python train.py --experiment mixed_optimization --model_type s

# ZeRO-3
python train.py --experiment zero3 --model_type s

# 消融实验
python train.py --experiment ablation --model_type s
```

### 使用YOLOv5m模型

```bash
python train.py --experiment baseline --model_type m --batch_size 4
```

### 批量运行所有实验

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh s  # 使用YOLOv5s
```

**Windows:**
```bash
run_all.bat s
```

### 自定义配置

```bash
python train.py \
    --experiment baseline \
    --model_type s \
    --data_path ./data/my_dataset \
    --epochs 100 \
    --batch_size 16 \
    --img_size 640 \
    --output_dir ./outputs/my_experiment
```

## 配置参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment` | required | 实验类型 (baseline/mixed_precision/gradient_checkpoint/mixed_optimization/zero3/ablation) |
| `--model_type` | s | YOLOv5模型类型 (s/m) |
| `--data_path` | ./data/kitti_subset | 数据集路径 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 (YOLOv5s: 8, YOLOv5m: 4) |
| `--img_size` | 640 | 图像尺寸 |
| `--config` | null | 配置文件路径 (JSON格式) |
| `--output_dir` | ./outputs | 输出目录 |

### 超参数 (在config文件中配置)

```json
{
  "hyp": {
    "lr0": 0.01,              # 初始学习率
    "lrf": 0.01,              # 学习率衰减因子
    "momentum": 0.937,        # SGD动量
    "weight_decay": 0.0005,   # 权重衰减
    "warmup_epochs": 3.0,     # 预热轮数
    "warmup_momentum": 0.8,   # 预热动量
    "warmup_bias_lr": 0.1,    # 预热偏置学习率
    "box": 0.05,              # 边界框损失权重
    "cls": 0.5,               # 类别损失权重
    "obj": 1.0                # 目标性损失权重
  }
}
```

## 输出结果

### 训练输出

每个实验会生成：

```
outputs/<experiment_name>/
├── training.log           # 训练日志
├── <experiment>_results.json   # 结果摘要
└── <experiment>_model.pt       # 训练好的模型
```

### 结果文件格式

`results.json` 包含以下指标：

```json
{
    "experiment": "baseline",
    "model_type": "s",
    "peak_mem_GiB": 8.42,      // 峰值显存占用 (GiB)
    "avg_iter_time_s": 0.123,  // 平均迭代时间 (秒)
    "final_map": 45.67         // 最终 mAP@0.5 (%)
}
```

### 对比实验结果

运行所有实验后，生成对比报告：

```bash
python scripts/compare_experiments.py --output_dir ./outputs
```

生成文件：
- `comparison_report.txt` - 文本报告
- `comparison_results.json` - JSON结果
- `comparison_plot.png` - 对比图表
- `comparison_plot.pdf` - PDF图表

## 优化方案说明

### 1. 基准组 (Baseline)
- **精度**: FP32
- **特点**: 无任何优化，作为对比基准
- **适用**: 小模型、小batch size

### 2. 混合精度 (Mixed Precision)
- **精度**: FP16 (自动混合精度)
- **特点**:
  - 显存占用降低约40-50%
  - 训练速度提升约20-30%
  - 需要梯度缩放保证数值稳定性
- **适用**: 大多数场景

### 3. 梯度检查点 (Gradient Checkpoint)
- **精度**: FP32
- **特点**:
  - 显存占用降低约30-40%
  - 训练速度降低约10-20%（需重新计算前向传播）
  - 每4层保存一次激活值
- **适用**: 显存受限场景

### 4. 混合优化 (Mixed + Checkpoint)
- **精度**: FP16 + 梯度检查点
- **特点**:
  - 显存占用降低约60-70%
  - 训练速度略有提升
  - 结合两者优势
- **适用**: 大模型训练

### 5. ZeRO-3 (DeepSpeed)
- **精度**: FP16 + ZeRO-3
- **特点**:
  - 支持4卡并行
  - 优化器状态和参数可卸载到CPU
  - 显存占用最低
  - 需要多GPU环境
- **适用**: 超大模型训练

### 6. 消融实验
- **精度**: FP16 + 每2层检查点
- **特点**:
  - 更密集的检查点设置
  - 显存优化效果更好
  - 计算开销更大
- **适用**: 极限显存优化

## 性能预期

基于RTX 3090, YOLOv5s, batch_size=8:

| 方案 | 显存占用 | 迭代时间 | mAP |
|------|----------|----------|-----|
| 基准组 | ~8-9 GiB | 0.12s | 45-50% |
| 混合精度 | ~4-5 GiB | 0.09s | 45-50% |
| 梯度检查点 | ~5-6 GiB | 0.14s | 45-50% |
| 混合优化 | ~3-4 GiB | 0.10s | 45-50% |
| ZeRO-3 | ~2-3 GiB/GPU | 0.08s | 45-50% |
| 消融实验 | ~2-3 GiB | 0.13s | 45-50% |

## 常见问题

### 1. CUDA Out of Memory

**解决方案**:
- 减小 `batch_size`
- 使用梯度检查点或混合精度
- 减小 `img_size`
- 使用 ZeRO-3

### 2. DeepSpeed 初始化失败

**解决方案**:
```bash
# 安装正确的DeepSpeed版本
pip install deepspeed==0.5.10

# 检查CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 3. 数据加载缓慢

**解决方案**:
- 增加 `workers` 参数
- 使用SSD存储数据集
- 启用 `pin_memory=True`

### 4. YOLOv5 导入失败

**解决方案**:
```bash
# 克隆YOLOv5源码
git clone https://github.com/ultralytics/yolov5.git

# 将yolov5目录放在项目根目录
# 或者修改 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./yolov5
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{yolov5_training_optimization,
  title={YOLOv5 Training Optimization Experiments},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/yolov5_training_optimization}
}
```

## 许可证

本项目基于 MIT 许可证开源。

## 联系方式

- 项目主页: [GitHub URL]
- 问题反馈: [Issues]
- 邮箱: your.email@example.com

## 更新日志

### v1.0.0 (2024-12-25)
- 初始版本发布
- 实现6种优化方案
- 添加数据集准备脚本
- 添加实验对比工具
