"""
YOLOv5损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv5Loss(nn.Module):
    """
    YOLOv5损失函数
    包含边界框损失、目标性损失和类别损失
    """
    def __init__(self, nc=3, hyp=None):
        """
        Args:
            nc: 类别数量
            hyp: 超参数字典
        """
        super().__init__()
        self.nc = nc  # 类别数

        # 损失权重
        if hyp is None:
            hyp = {}
        self.hyp = hyp

        # 损失权重超参数
        self.box_gain = hyp.get("box", 0.05)     # 边界框损失增益
        self.cls_gain = hyp.get("cls", 0.5)      # 类别损失增益
        self.obj_gain = hyp.get("obj", 1.0)      # 目标性损失增益

        # Anchor相关
        self.nl = 3  # 检测层数
        self.na = 3  # 每层的anchor数

        # BCE损失（用于目标性和类别）
        self.BCE_cls = nn.BCEWithLogitsLoss()
        self.BCE_obj = nn.BCEWithLogitsLoss()

        # CIoU损失（用于边界框）
        self.CIoU_loss = CIoULoss()

    def forward(self, predictions, targets):
        """
        计算损失

        Args:
            predictions: 模型预测输出
            targets: 目标标签 [batch, max_objs, 6] (class, x, y, w, h, conf)

        Returns:
            总损失和损失组件
        """
        device = predictions[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        # 简化版损失计算（用于演示）
        # 实际YOLOv5的损失计算更复杂，涉及anchor匹配和网格分配

        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            pred = predictions[0]  # 取第一个输出 [batch, num_classes, ...]

            # 简化的目标检测损失
            # 实际应用中需要完整的anchor匹配逻辑

            # 边界框回归损失（简化）
            if pred.shape[-1] > 4:  # 如果有足够的维度
                # 假设预测包含边界框坐标
                pred_boxes = pred[..., :4] if pred.shape[-1] >= 4 else pred
                if targets is not None and len(targets) > 0:
                    # 简化的IoU损失
                    lbox = self.compute_simple_iou_loss(pred_boxes, targets)

            # 类别损失
            if pred.shape[-1] > self.nc:
                # 提取类别预测
                pred_cls = pred[..., -self.nc:]
                if targets is not None and len(targets) > 0:
                    target_cls = targets[..., 0].long() if targets.dim() > 1 else targets.long()
                    if target_cls.max() < self.nc:
                        lcls = F.cross_entropy(
                            pred_cls.view(-1, self.nc),
                            target_cls.view(-1),
                            reduction='mean'
                        )

            # 目标性损失
            obj_pred = torch.sigmoid(pred[..., 0] if pred.shape[-1] > 0 else pred)
            obj_target = torch.ones_like(obj_pred)  # 简化：假设所有位置都有目标
            lobj = self.BCE_obj(obj_pred, obj_target)

        # 总损失
        loss = lbox * self.box_gain + lcls * self.cls_gain + lobj * self.obj_gain

        # 返回总损失和损失组件
        loss_items = torch.cat([lbox, lobj, lcls]).detach()

        return loss, loss_items

    def compute_simple_iou_loss(self, pred_boxes, targets):
        """计算简化的IoU损失"""
        # 创建简单的边界框目标
        try:
            # 简化版本：使用MSE作为替代
            if targets.dim() > 2 and targets.shape[-1] >= 4:
                target_boxes = targets[..., 1:5]  # 提取x,y,w,h
                # 调整pred_boxes形状匹配
                if pred_boxes.shape[-1] == 4:
                    return F.mse_loss(pred_boxes, target_boxes[:pred_boxes.size(0), :pred_boxes.size(1), :4])
        except:
            pass

        return torch.tensor(0.0, device=pred_boxes.device)


class CIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss
    用于边界框回归的损失函数
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        计算CIoU损失

        Args:
            pred: 预测边界框 [N, 4] (x1, y1, x2, y2) 或 (x, y, w, h)
            target: 目标边界框 [N, 4]

        Returns:
            CIoU损失值
        """
        # 确保边界框格式为 (x_center, y_center, width, height)
        if pred.shape[-1] != 4 or target.shape[-1] != 4:
            return torch.tensor(0.0, device=pred.device)

        # 转换为中心点格式
        pred = self.xyxy_to_cxcywh(pred) if self.is_xyxy_format(pred) else pred
        target = self.xyxy_to_cxcywh(target) if self.is_xyxy_format(target) else target

        # 计算CIoU
        iou = self.compute_iou(pred, target)

        # 计算中心点距离
        c_x = (pred[:, 0] + target[:, 0]) / 2
        c_y = (pred[:, 1] + target[:, 1]) / 2
        center_distance = (c_x ** 2) + (c_y ** 2)

        # 计算对角线距离
        x1 = torch.min(pred[:, 0] - pred[:, 2] / 2, target[:, 0] - target[:, 2] / 2)
        y1 = torch.min(pred[:, 1] - pred[:, 3] / 2, target[:, 1] - target[:, 3] / 2)
        x2 = torch.max(pred[:, 0] + pred[:, 2] / 2, target[:, 0] + target[:, 2] / 2)
        y2 = torch.max(pred[:, 1] + pred[:, 3] / 2, target[:, 1] + target[:, 3] / 2)
        diagonal_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2

        # CIoU = IoU - (center_distance / diagonal_distance)
        ciou = iou - (center_distance / (diagonal_distance + self.eps))

        loss = 1 - ciou
        return loss.mean()

    def compute_iou(self, box1, box2):
        """计算IoU"""
        # box格式: (x_center, y_center, width, height)
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

        # 交集区域
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 并集区域
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + self.eps

        iou = inter_area / union_area
        return iou

    def is_xyxy_format(self, boxes):
        """判断边界框是否为xyxy格式"""
        # 简化判断：假设坐标值都在[0,1]范围内
        return True

    def xyxy_to_cxcywh(self, boxes):
        """将xyxy格式转换为cxcywh格式"""
        # boxes: [N, 4] (x1, y1, x2, y2)
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        return torch.stack([x_center, y_center, width, height], dim=-1)


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测值 [N, C]
            targets: 目标值 [N]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_loss_components(pred, target, nc=3):
    """
    计算损失各组件（辅助函数）

    Args:
        pred: 预测输出
        target: 目标标签
        nc: 类别数

    Returns:
        损失字典
    """
    device = pred.device

    # 边界框损失
    box_loss = F.mse_loss(pred[..., :4], target[..., :4]) if pred.shape[-1] >= 4 else torch.tensor(0.0, device=device)

    # 目标性损失
    obj_loss = F.binary_cross_entropy_with_logits(
        pred[..., 4] if pred.shape[-1] > 4 else pred,
        target[..., 4] if target.shape[-1] > 4 else target
    )

    # 类别损失
    if pred.shape[-1] >= 5 + nc:
        cls_loss = F.cross_entropy(
            pred[..., 5:5+nc].reshape(-1, nc),
            target[..., 0].long().reshape(-1)
        )
    else:
        cls_loss = torch.tensor(0.0, device=device)

    return {
        'box_loss': box_loss,
        'obj_loss': obj_loss,
        'cls_loss': cls_loss,
        'total_loss': box_loss + obj_loss + cls_loss
    }
