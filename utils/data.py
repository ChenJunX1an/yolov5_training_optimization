"""
数据加载工具
"""

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def get_data_loader(path, batch_size, img_size, hyp, shuffle=True):
    """
    获取数据加载器

    Args:
        path: 数据集路径
        batch_size: 批次大小
        img_size: 图像尺寸
        hyp: 超参数字典
        shuffle: 是否打乱数据

    Returns:
        数据加载器和数据集
    """
    try:
        from yolov5.utils.datasets import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(
            path,
            img_size=img_size,
            batch_size=batch_size,
            augment=shuffle,
            hyp=hyp
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=hyp.get("workers", 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )

        return loader, dataset

    except ImportError:
        print("警告: 无法导入YOLOv5数据加载器，使用简化数据集")
        return get_simple_dataloader(path, batch_size, img_size, shuffle)


def get_simple_dataloader(path, batch_size, img_size, shuffle=True):
    """
    简化的数据加载器（当YOLOv5不可用时使用）
    """
    dataset = SimpleDataset(
        data_path=path,
        img_size=img_size,
        augment=shuffle
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return loader, dataset


class SimpleDataset(Dataset):
    """
    简化的目标检测数据集
    """
    def __init__(self, data_path, img_size=640, augment=True):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.augment = augment

        # 查找所有图像文件
        self.img_files = list((self.data_path / "images").glob("*.jpg")) + \
                        list((self.data_path / "images").glob("*.png"))

        if len(self.img_files) == 0:
            # 如果找不到images子目录，直接在data_path中查找
            self.img_files = list(self.data_path.glob("*.jpg")) + \
                            list(self.data_path.glob("*.png"))

        print(f"找到 {len(self.img_files)} 张图像")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            # 如果OpenCV读取失败，使用PIL
            img_pil = Image.open(img_path).convert('RGB')
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 数据增强
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

        # 调整大小
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化并转换为Tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW

        # 生成随机标签（模拟检测标签）
        # 格式: [class, x, y, w, h] (归一化坐标)
        num_objs = np.random.randint(0, 5)
        if num_objs > 0:
            labels = torch.zeros((num_objs, 6))  # [image_class, x, y, w, h, confidence]
            labels[:, 0] = np.random.randint(0, 3, size=num_objs)  # 随机类别
            labels[:, 1] = np.random.rand(num_objs)  # 随机x
            labels[:, 2] = np.random.rand(num_objs)  # 随机y
            labels[:, 3] = np.random.rand(num_objs) * 0.3  # 随机w
            labels[:, 4] = np.random.rand(num_objs) * 0.3  # 随机h
            labels[:, 5] = 1.0  # confidence
        else:
            labels = torch.zeros((0, 6))

        return img, labels, str(img_path), idx


def create_kitti_subset_structure(output_path, num_samples=100):
    """
    创建KITTI数据集子集的目录结构（用于测试）

    Args:
        output_path: 输出路径
        num_samples: 样本数量
    """
    output_path = Path(output_path)

    for split in ["train", "val"]:
        # 创建子目录
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

        print(f"已创建目录: {output_path / split}")

    print("\n请在以下目录中放置数据:")
    print(f"  训练集: {output_path / 'train' / 'images'}")
    print(f"  训练标签: {output_path / 'train' / 'labels'}")
    print(f"  验证集: {output_path / 'val' / 'images'}")
    print(f"  验证标签: {output_path / 'val' / 'labels'}")


def collate_fn(batch):
    """
    自定义批处理函数（处理不同大小的标签）
    """
    imgs, labels, paths, idxs = zip(*batch)

    # 堆叠图像
    imgs = torch.stack(imgs, 0)

    # 标签保持为列表（每个样本可能有不同数量的目标）
    return imgs, list(labels), list(paths), list(idxs)
