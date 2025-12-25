"""
YOLOv5 Training Optimization Experiments
主训练脚本 - 包含所有实验方案
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint_sequential
import deepspeed
import time
import numpy as np
import argparse
import os
import json
from pathlib import Path

# 导入工具模块
from utils.metrics import MetricTracker, compute_map
from utils.model import get_yolov5_model
from utils.data import get_data_loader
from utils.loss import YOLOv5Loss

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============= 实验方案1: 基准组（无优化，FP32）=============
def train_baseline(config):
    """基准组实验 - FP32精度训练"""
    print("\n" + "="*50)
    print("开始基准组实验（FP32）")
    print("="*50)

    # 初始化组件
    model = get_yolov5_model(model_type=config["model_type"])
    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["hyp"]["lr0"],
        weight_decay=config["hyp"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()

    # 训练循环
    model.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device, dtype=torch.float32)  # FP32精度
            targets = targets.to(device)

            # 前向+反向+优化
            outputs = model(imgs)
            loss, loss_items = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.end_iter()

            # 每100迭代打印一次
            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        scheduler.step()

        # 每个epoch结束后验证
        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    # 计算最终mAP
    final_map = compute_map(model, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "baseline",
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\n基准组实验结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model, results


# ============= 实验方案2: 混合精度组（FP16）=============
def train_mixed_precision(config):
    """混合精度实验 - FP16训练"""
    print("\n" + "="*50)
    print("开始混合精度组实验（FP16）")
    print("="*50)

    model = get_yolov5_model(model_type=config["model_type"])
    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["hyp"]["lr0"],
        weight_decay=config["hyp"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()

    # 混合精度核心组件
    scaler = GradScaler()

    model.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device, dtype=torch.float16)  # FP16精度
            targets = targets.to(device)

            # 自动混合精度前向传播
            with autocast():
                outputs = model(imgs)
                loss, loss_items = criterion(outputs, targets)

            # 反向传播（缩放梯度）
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tracker.end_iter()

            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    final_map = compute_map(model, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "mixed_precision",
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\n混合精度组实验结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model, results


# ============= 实验方案3: 梯度检查点组（每4层）=============
def train_gradient_checkpoint(config):
    """梯度检查点实验 - 每4层设置检查点"""
    print("\n" + "="*50)
    print("开始梯度检查点组实验（每4层）")
    print("="*50)

    model = get_yolov5_model(model_type=config["model_type"])
    checkpoint_interval = config.get("checkpoint_interval", 4)

    # 对YOLOv5主干网络应用梯度检查点
    if hasattr(model.model[0], 'gradient_checkpointing'):
        model.model[0].gradient_checkpointing = True

    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["hyp"]["lr0"],
        weight_decay=config["hyp"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()

    model.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device, dtype=torch.float32)
            targets = targets.to(device)

            # 梯度检查点前向传播
            outputs = model(imgs)
            loss, loss_items = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.end_iter()

            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    final_map = compute_map(model, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "gradient_checkpoint",
        "checkpoint_interval": checkpoint_interval,
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\n梯度检查点组实验结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model, results


# ============= 实验方案4: 混合优化组（FP16 + 每4层检查点）=============
def train_mixed_optimization(config):
    """混合优化实验 - FP16 + 梯度检查点"""
    print("\n" + "="*50)
    print("开始混合优化组实验（FP16 + 每4层检查点）")
    print("="*50)

    model = get_yolov5_model(model_type=config["model_type"])
    checkpoint_interval = config.get("checkpoint_interval", 4)

    if hasattr(model.model[0], 'gradient_checkpointing'):
        model.model[0].gradient_checkpointing = True

    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["hyp"]["lr0"],
        weight_decay=config["hyp"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()
    scaler = GradScaler()

    model.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device, dtype=torch.float16)
            targets = targets.to(device)

            with autocast():
                outputs = model(imgs)
                loss, loss_items = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tracker.end_iter()

            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    final_map = compute_map(model, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "mixed_optimization",
        "checkpoint_interval": checkpoint_interval,
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\n混合优化组实验结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model, results


# ============= 实验方案5: ZeRO-3组（DeepSpeed优化）=============
def train_zero3(config):
    """ZeRO-3实验 - DeepSpeed优化"""
    print("\n" + "="*50)
    print("开始ZeRO-3组实验（DeepSpeed）")
    print("="*50)

    # DeepSpeed配置
    ds_config = {
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 65536,
            "stage3_prefetch_bucket_size": 65536,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 1e7,
            "stage3_max_reuse_distance": 1e7,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config["hyp"]["lr0"],
                "weight_decay": config["hyp"]["weight_decay"]
            }
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "params": {"T_max": config["epochs"]}
        },
        "train_micro_batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
    }

    model = get_yolov5_model(model_type=config["model_type"])
    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    # 初始化DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        training_data=train_loader.dataset,
        config_params=ds_config,
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()

    model_engine.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model_engine(imgs)
            loss, loss_items = criterion(outputs, targets)

            model_engine.backward(loss)
            model_engine.step()

            tracker.end_iter()

            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model_engine.module, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    final_map = compute_map(model_engine.module, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "zero3",
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\nZeRO-3组实验结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model_engine, results


# ============= 实验方案6: 消融实验组（每2层检查点）=============
def train_ablation_experiment(config):
    """消融实验 - 每2层检查点"""
    print("\n" + "="*50)
    print("开始消融实验组（FP16 + 每2层检查点）")
    print("="*50)

    model = get_yolov5_model(model_type=config["model_type"])
    config["checkpoint_interval"] = 2  # 每2层

    if hasattr(model.model[0], 'gradient_checkpointing'):
        model.model[0].gradient_checkpointing = True

    train_loader, train_dataset = get_data_loader(
        config["train_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"]
    )
    val_loader, val_dataset = get_data_loader(
        config["val_path"],
        config["batch_size"],
        config["img_size"],
        config["hyp"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["hyp"]["lr0"],
        weight_decay=config["hyp"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    criterion = YOLOv5Loss(nc=config["nc"], hyp=config["hyp"])
    tracker = MetricTracker()
    scaler = GradScaler()

    model.train()
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            tracker.start_iter()

            imgs = imgs.to(device, dtype=torch.float16)
            targets = targets.to(device)

            with autocast():
                outputs = model(imgs)
                loss, loss_items = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tracker.end_iter()

            if (i+1) % 100 == 0:
                stats = tracker.get_stats()
                print(f"Iter [{i+1}], Loss: {loss.item():.4f}, "
                      f"Peak Mem: {stats['peak_mem_GiB']} GiB, "
                      f"Avg Time: {stats['avg_iter_time_s']}s")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            current_map = compute_map(model, val_loader, device, config)
            print(f"Epoch {epoch+1} mAP@0.5: {current_map}%")

    final_map = compute_map(model, val_loader, device, config)
    stats = tracker.get_stats()

    results = {
        "experiment": "ablation_2layers",
        "checkpoint_interval": 2,
        "model_type": config["model_type"],
        "peak_mem_GiB": stats['peak_mem_GiB'],
        "avg_iter_time_s": stats['avg_iter_time_s'],
        "final_map": final_map
    }

    print("\n消融实验组结果：")
    print(f"峰值显存占用：{stats['peak_mem_GiB']} GiB")
    print(f"平均迭代时间：{stats['avg_iter_time_s']} s")
    print(f"验证集mAP@0.5：{final_map}%")

    return model, results


# ============= 主函数 =============
def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Training Optimization")
    parser.add_argument("--experiment", type=str, default="baseline",
                       choices=["baseline", "mixed_precision", "gradient_checkpoint",
                               "mixed_optimization", "zero3", "ablation"],
                       help="选择实验方案")
    parser.add_argument("--model_type", type=str, default="s", choices=["s", "m"],
                       help="YOLOv5模型类型")
    parser.add_argument("--data_path", type=str, default="./data/kitti_subset",
                       help="数据集路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--img_size", type=int, default=640, help="图像尺寸")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（JSON格式）")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")

    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            "train_path": f"{args.data_path}/train",
            "val_path": f"{args.data_path}/val",
            "nc": 3,  # 类别数
            "hyp": {
                "lr0": 0.01,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "momentum": 0.937,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1
            },
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "model_type": args.model_type,
            "workers": 4
        }

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行选定的实验
    experiment_map = {
        "baseline": train_baseline,
        "mixed_precision": train_mixed_precision,
        "gradient_checkpoint": train_gradient_checkpoint,
        "mixed_optimization": train_mixed_optimization,
        "zero3": train_zero3,
        "ablation": train_ablation_experiment
    }

    train_func = experiment_map[args.experiment]
    model, results = train_func(config)

    # 保存结果
    results_file = output_dir / f"{args.experiment}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # 保存模型
    model_file = output_dir / f"{args.experiment}_model.pt"
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), model_file)
    else:
        torch.save(model.state_dict(), model_file)

    print(f"\n实验完成！结果已保存至: {results_file}")
    print(f"模型已保存至: {model_file}")


if __name__ == "__main__":
    main()
