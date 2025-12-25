"""
模型定义和加载工具
"""

import torch
import torch.nn as nn
from pathlib import Path


def get_yolov5_model(model_type="s", nc=3, pretrained=True):
    """
    获取YOLOv5模型

    Args:
        model_type: 模型类型 ("s" 或 "m")
        nc: 类别数量
        pretrained: 是否使用预训练权重

    Returns:
        YOLOv5模型
    """
    try:
        from yolov5.models.yolo import Model

        cfg = f"yolov5/models/yolov5{model_type}.yaml"
        model = Model(cfg, ch=3, nc=nc)

        # 加载预训练权重（如果需要）
        if pretrained:
            weights_path = f"yolov5/weights/yolov5{model_type}.pt"
            if Path(weights_path).exists():
                checkpoint = torch.load(weights_path, map_location='cpu')
                # 过滤掉类别数不匹配的层
                state_dict = checkpoint['model'].state_dict() if 'model' in checkpoint else checkpoint
                model.load_state_dict(state_dict, strict=False)
                print(f"已加载预训练权重: {weights_path}")
            else:
                print(f"警告: 预训练权重文件不存在: {weights_path}")

        return model

    except ImportError:
        print("警告: 无法导入YOLOv5，使用简化模型")
        return get_simple_cnn(nc=nc)


def get_simple_cnn(nc=3):
    """
    简化的CNN模型（当YOLOv5不可用时使用）
    """
    class SimpleDetector(nn.Module):
        def __init__(self, num_classes=nc):
            super().__init__()
            # 特征提取 backbone
            self.features = nn.Sequential(
                # Conv1
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Conv2
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # Conv3
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # Conv4
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

            # 检测头
            self.detect = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            features = self.features(x)
            output = self.detect(features)
            return [output]

    return SimpleDetector(nc)


def freeze_layers(model, num_layers_to_freeze=0):
    """
    冻结模型的前N层

    Args:
        model: PyTorch模型
        num_layers_to_freeze: 要冻结的层数
    """
    if num_layers_to_freeze <= 0:
        return

    layers_frozen = 0
    for name, param in model.named_parameters():
        if layers_frozen < num_layers_to_freeze:
            param.requires_grad = False
            layers_frozen += 1
        else:
            break

    print(f"已冻结 {layers_frozen} 层参数")


def unfreeze_layers(model):
    """
    解冻所有层
    """
    for param in model.parameters():
        param.requires_grad = True
    print("已解冻所有层参数")


def get_model_info(model, input_size=(3, 640, 640)):
    """
    获取模型信息

    Args:
        model: PyTorch模型
        input_size: 输入尺寸

    Returns:
        模型信息字典
    """
    from ..utils.metrics import count_parameters, get_model_size

    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)

    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size,
        "input_size": input_size
    }

    # 尝试计算FLOPs
    try:
        from thop import profile
        dummy_input = torch.randn(1, *input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        info["flops"] = f"{flops / 1e9:.2f}G"
    except:
        info["flops"] = "N/A"

    return info


def print_model_summary(model, input_size=(3, 640, 640)):
    """
    打印模型摘要信息

    Args:
        model: PyTorch模型
        input_size: 输入尺寸
    """
    info = get_model_info(model, input_size)

    print("\n" + "="*50)
    print("模型摘要")
    print("="*50)
    print(f"总参数数: {info['total_parameters']:,}")
    print(f"可训练参数: {info['trainable_parameters']:,}")
    print(f"模型大小: {info['model_size_mb']} MB")
    print(f"FLOPs: {info['flops']}")
    print("="*50 + "\n")
