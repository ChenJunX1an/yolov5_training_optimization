"""
指标统计和评估工具
"""

import time
import numpy as np
import torch
from pathlib import Path


class MetricTracker:
    """训练指标追踪器"""
    def __init__(self):
        self.start_time = 0
        self.iter_times = []
        self.memory_usage = []

    def start_iter(self):
        """记录迭代开始"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            # 记录当前显存占用（GB）
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.memory_usage.append(mem)

    def end_iter(self):
        """记录迭代结束"""
        iter_time = time.time() - self.start_time
        self.iter_times.append(iter_time)

    def get_stats(self):
        """获取统计信息"""
        if len(self.iter_times) <= 10:
            avg_time = np.mean(self.iter_times) if self.iter_times else 0
        else:
            avg_time = np.mean(self.iter_times[10:])  # 跳过前10次预热迭代

        peak_mem = max(self.memory_usage) if self.memory_usage else 0
        return {
            "peak_mem_GiB": round(peak_mem, 2),
            "avg_iter_time_s": round(avg_time, 3)
        }

    def reset(self):
        """重置统计"""
        self.start_time = 0
        self.iter_times = []
        self.memory_usage = []


def compute_map(model, val_loader, device, config):
    """
    计算mAP指标

    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 设备
        config: 配置字典

    Returns:
        mAP值（百分比）
    """
    model.eval()

    try:
        from yolov5.utils.metrics import ap_per_class
        from yolov5.utils.general import non_max_suppression, scale_coords
    except ImportError:
        print("警告: 无法导入YOLOv5评估模块，返回默认值")
        return 0.0

    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, targets, paths, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            # 转换为检测格式（x1,y1,x2,y2,conf,cls）
            preds = non_max_suppression(
                outputs[0],
                conf_thres=0.001,
                iou_thres=0.65
            )

            for i, pred in enumerate(preds):
                if pred is not None and len(pred) > 0:
                    # 缩放坐标到原始图像尺寸
                    pred[:, :4] = scale_coords(
                        imgs.shape[2:],
                        pred[:, :4],
                        targets[i][:, 2:6].shape if len(targets) > i and targets[i] is not None else imgs.shape[2:]
                    )
                    all_preds.append(pred.cpu().numpy())

                if i < len(targets) and targets[i] is not None:
                    all_targets.append(targets[i][:, 1:].cpu().numpy())  # [cls, x1,y1,x2,y2]

    # 计算mAP@0.5
    if len(all_preds) > 0 and len(all_targets) > 0:
        try:
            ap, f1, _, _ = ap_per_class(
                all_preds,
                all_targets,
                plot=False,
                save_dir=".",
                names=[f"cls{i}" for i in range(config["nc"])]
            )
            mAP = ap.mean() * 100
            return round(mAP, 2)
        except Exception as e:
            print(f"mAP计算错误: {e}")
            return 0.0
    else:
        return 0.0


class ValidationLogger:
    """验证日志记录器"""
    def __init__(self, save_dir="./logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []

    def log(self, epoch, metrics, model_name="model"):
        """记录指标"""
        log_entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            "metrics": metrics
        }
        self.metrics_history.append(log_entry)

        # 打印日志
        print(f"\nEpoch {epoch} Validation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    def save_history(self, filename="validation_history.json"):
        """保存历史记录"""
        import json
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        print(f"\n验证历史已保存至: {save_path}")


def calculate_inference_time(model, input_size=(1, 3, 640, 640), device="cuda", num_runs=100):
    """
    计算模型推理时间

    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
        num_runs: 运行次数

    Returns:
        平均推理时间（毫秒）
    """
    model.eval()

    # 创建随机输入
    dummy_input = torch.randn(input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    return round(avg_time, 3)


def get_model_size(model):
    """
    获取模型大小（参数量）

    Args:
        model: PyTorch模型

    Returns:
        模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_mb, 2)


def count_parameters(model):
    """
    统计模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        总参数数和可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
