# 项目文件说明

## 核心文件

### train.py
主训练脚本，包含6种实验方案：
- `train_baseline()` - 基准组（FP32）
- `train_mixed_precision()` - 混合精度（FP16）
- `train_gradient_checkpoint()` - 梯度检查点
- `train_mixed_optimization()` - 混合优化
- `train_zero3()` - ZeRO-3
- `train_ablation_experiment()` - 消融实验

## 工具模块 (utils/)

### __init__.py
模块导出定义

### data.py
数据加载工具
- `get_data_loader()` - 获取YOLOv5数据加载器
- `SimpleDataset` - 简化数据集（YOLOv5不可用时）
- `create_kitti_subset_structure()` - 创建数据集目录
- `collate_fn()` - 批处理函数

### metrics.py
指标统计和评估
- `MetricTracker` - 训练指标追踪器
- `compute_map()` - 计算mAP
- `ValidationLogger` - 验证日志记录器
- `calculate_inference_time()` - 推理时间计算
- `get_model_size()` - 模型大小计算
- `count_parameters()` - 参数数量统计

### model.py
模型定义和加载
- `get_yolov5_model()` - 获取YOLOv5模型
- `get_simple_cnn()` - 简化CNN模型
- `freeze_layers()` - 冻结层
- `unfreeze_layers()` - 解冻层
- `get_model_info()` - 获取模型信息
- `print_model_summary()` - 打印模型摘要

### loss.py
损失函数
- `YOLOv5Loss` - YOLOv5损失函数
- `CIoULoss` - CIoU损失
- `FocalLoss` - Focal损失
- `compute_loss_components()` - 损失组件计算

## 配置文件 (configs/)

所有配置文件使用JSON格式，包含：
- `experiment_name` - 实验名称
- `description` - 实验描述
- `data` - 数据配置（路径、类别数、图像尺寸）
- `model` - 模型配置（类型、预训练）
- `training` - 训练配置（轮数、批次、线程数）
- `hyp` - 超参数（学习率、权重衰减等）
- `optimization` - 优化配置（混合精度、检查点、ZeRO）
- `deepspeed` - DeepSpeed配置（仅zero3）

## 辅助脚本 (scripts/)

### prepare_dataset.py
数据集准备工具
```bash
# 创建目录结构
python scripts/prepare_dataset.py --mode create --output ./data/kitti_subset

# 分割数据集
python scripts/prepare_dataset.py --mode split --source ./data/kitti --output ./data/kitti_subset

# 验证数据集
python scripts/prepare_dataset.py --mode verify --output ./data/kitti_subset
```

### compare_experiments.py
实验结果对比工具
```bash
python scripts/compare_experiments.py --output_dir ./outputs
```

## 运行脚本

### run_all.sh / run_all.bat
批量运行所有实验

### run_baseline.sh / run_baseline.bat
运行基准组实验

## 输出结构

```
outputs/
├── baseline/
│   ├── training.log
│   ├── baseline_results.json
│   └── baseline_model.pt
├── mixed_precision/
│   ├── training.log
│   ├── mixed_precision_results.json
│   └── mixed_precision_model.pt
├── ...
├── comparison_report.txt
├── comparison_results.json
├── comparison_plot.png
└── comparison_plot.pdf
```

## 数据集结构

```
data/kitti_subset/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   └── ...
│   └── labels/
│       ├── 000001.txt
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

## 扩展指南

### 添加新的实验方案

1. 在 `train.py` 中添加训练函数
2. 在 `configs/` 中创建配置文件
3. 在 `train.py:main()` 的 `experiment_map` 中注册

### 修改损失函数

编辑 `utils/loss.py` 中的 `YOLOv5Loss` 类

### 自定义数据增强

编辑 `utils/data.py` 中的 `SimpleDataset` 类
