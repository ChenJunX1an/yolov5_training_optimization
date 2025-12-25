"""
数据集准备脚本
创建KITTI子集目录结构
"""

import argparse
from pathlib import Path
import shutil
import random


def create_kitti_subset_structure(output_path, num_samples=100):
    """
    创建KITTI数据集子集的目录结构

    Args:
        output_path: 输出路径
        num_samples: 每个split的样本数量
    """
    output_path = Path(output_path)

    print("创建KITTI子集目录结构...")
    print(f"输出路径: {output_path}")
    print(f"样本数量: {num_samples}")
    print("-" * 50)

    for split in ["train", "val"]:
        # 创建子目录
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

        print(f"✓ 已创建目录: {output_path / split}")

    print("-" * 50)
    print("\n数据集目录结构:")
    print(f"  训练集图像: {output_path / 'train' / 'images'}")
    print(f"  训练集标签: {output_path / 'train' / 'labels'}")
    print(f"  验证集图像: {output_path / 'val' / 'images'}")
    print(f"  验证集标签: {output_path / 'val' / 'labels'}")
    print("\n请将数据放入上述目录")
    print("  - 图像格式: .jpg, .png")
    print("  - 标签格式: YOLO格式 (class x_center y_center width height)，归一化到[0,1]")


def split_dataset(source_path, output_path, val_ratio=0.2, seed=42):
    """
    将数据集分割为训练集和验证集

    Args:
        source_path: 源数据路径（包含images和labels文件夹）
        output_path: 输出路径
        val_ratio: 验证集比例
        seed: 随机种子
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    random.seed(seed)

    # 获取所有图像文件
    img_dir = source_path / "images"
    if not img_dir.exists():
        print(f"错误: 图像目录不存在: {img_dir}")
        return

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"找到 {len(img_files)} 张图像")

    # 随机打乱
    random.shuffle(img_files)

    # 分割
    split_idx = int(len(img_files) * (1 - val_ratio))
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")

    # 创建目标目录
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # 复制文件
    def copy_files(files, split):
        for img_file in files:
            # 复制图像
            dst_img = output_path / split / "images" / img_file.name
            shutil.copy2(img_file, dst_img)

            # 复制对应的标签文件
            label_file = img_file.stem + ".txt"
            src_label = source_path / "labels" / label_file
            if src_label.exists():
                dst_label = output_path / split / "labels" / label_file
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 找不到标签文件 {label_file}")

    print("\n复制训练集文件...")
    copy_files(train_files, "train")

    print("复制验证集文件...")
    copy_files(val_files, "val")

    print("\n✓ 数据集分割完成!")


def verify_dataset(dataset_path):
    """
    验证数据集完整性

    Args:
        dataset_path: 数据集路径
    """
    dataset_path = Path(dataset_path)

    print("\n验证数据集完整性...")
    print("=" * 50)

    for split in ["train", "val"]:
        img_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            print(f"✗ {split} 图像目录不存在")
            continue

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        label_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        print(f"\n{split.upper()}集:")
        print(f"  图像数量: {len(img_files)}")
        print(f"  标签数量: {len(label_files)}")

        # 检查一一对应
        missing_labels = []
        for img_file in img_files:
            label_file = label_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                missing_labels.append(img_file.name)

        if missing_labels:
            print(f"  缺失标签: {len(missing_labels)} 个")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    print(f"    - {name}")
        else:
            print(f"  ✓ 所有图像都有对应的标签")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="准备YOLOv5训练数据集")
    parser.add_argument("--mode", type=str, default="create",
                       choices=["create", "split", "verify"],
                       help="操作模式: create(创建目录), split(分割数据集), verify(验证)")
    parser.add_argument("--source", type=str, default="./data/kitti",
                       help="源数据路径")
    parser.add_argument("--output", type=str, default="./data/kitti_subset",
                       help="输出数据路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="创建目录时的样本数量")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                       help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    if args.mode == "create":
        create_kitti_subset_structure(args.output, args.num_samples)

    elif args.mode == "split":
        split_dataset(args.source, args.output, args.val_ratio, args.seed)
        verify_dataset(args.output)

    elif args.mode == "verify":
        verify_dataset(args.output)


if __name__ == "__main__":
    main()
