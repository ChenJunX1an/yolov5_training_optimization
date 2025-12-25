"""
实验结果对比脚本
比较不同实验方案的性能指标
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(output_dir):
    """
    加载所有实验结果

    Args:
        output_dir: 输出目录

    Returns:
        结果字典
    """
    output_dir = Path(output_dir)
    results = {}

    # 查找所有results文件
    for results_file in output_dir.glob("*/baseline_results.json"):
        exp_dir = results_file.parent
        exp_name = exp_dir.name

        try:
            with open(results_file, 'r') as f:
                results[exp_name] = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载 {results_file}: {e}")

    # 尝试从各个子目录加载
    for exp_dir in output_dir.iterdir():
        if exp_dir.is_dir():
            results_file = list(exp_dir.glob("*_results.json"))
            if results_file:
                try:
                    with open(results_file[0], 'r') as f:
                        results[exp_dir.name] = json.load(f)
                except Exception as e:
                    print(f"警告: 无法加载 {results_file[0]}: {e}")

    return results


def create_comparison_table(results):
    """
    创建对比表格

    Args:
        results: 结果字典

    Returns:
        pandas DataFrame
    """
    data = []
    for exp_name, exp_results in results.items():
        row = {
            "实验": exp_name,
            "模型": exp_results.get("model_type", "N/A"),
            "显存占用 (GiB)": exp_results.get("peak_mem_GiB", 0),
            "迭代时间 (s)": exp_results.get("avg_iter_time_s", 0),
            "mAP@0.5 (%)": exp_results.get("final_map", 0),
        }

        # 添加优化标志
        if "checkpoint_interval" in exp_results:
            row["检查点间隔"] = exp_results["checkpoint_interval"]
        else:
            row["检查点间隔"] = "N/A"

        data.append(row)

    df = pd.DataFrame(data)

    # 按实验名称排序
    if not df.empty:
        df = df.sort_values("实验")

    return df


def print_comparison(df):
    """
    打印对比表格

    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 80)
    print("实验结果对比")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    # 计算优化率
    if len(df) > 1:
        baseline = df[df["实验"] == "baseline"]
        if not baseline.empty:
            baseline_mem = baseline.iloc[0]["显存占用 (GiB)"]
            baseline_time = baseline.iloc[0]["迭代时间 (s)"]

            print("相比基准组的优化:")
            print("-" * 80)

            for _, row in df.iterrows():
                if row["实验"] != "baseline":
                    mem_reduction = ((baseline_mem - row["显存占用 (GiB)"]) / baseline_mem * 100)
                    time_change = ((baseline_time - row["迭代时间 (s)"]) / baseline_time * 100)

                    print(f"{row['实验']}:")
                    print(f"  显存优化: {mem_reduction:+.1f}%")
                    print(f"  速度变化: {time_change:+.1f}%")
                    print(f"  mAP变化: {row['mAP@0.5 (%)'] - baseline.iloc[0]['mAP@0.5 (%)']:+.2f}%")
            print("=" * 80 + "\n")


def plot_comparison(df, output_path):
    """
    绘制对比图表

    Args:
        df: pandas DataFrame
        output_path: 输出路径
    """
    output_path = Path(output_path)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv5训练优化实验对比', fontsize=16, fontweight='bold')

    # 1. 显存占用对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df["实验"], df["显存占用 (GiB)"], color='steelblue')
    ax1.set_title('峰值显存占用', fontweight='bold')
    ax1.set_ylabel('显存 (GiB)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # 2. 迭代时间对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df["实验"], df["迭代时间 (s)"], color='coral')
    ax2.set_title('平均迭代时间', fontweight='bold')
    ax2.set_ylabel('时间 (秒)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # 3. mAP对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df["实验"], df["mAP@0.5 (%)"], color='mediumseagreen')
    ax3.set_title('验证集mAP@0.5', fontweight='bold')
    ax3.set_ylabel('mAP (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 100])
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # 4. 综合性能雷达图
    ax4 = axes[1, 1]
    # 归一化指标
    metrics = ['显存效率', '速度', '精度']
    angles = [n / len(metrics) * 2 * 3.14159 for n in range(len(metrics))]
    angles += angles[:1]

    ax4 = plt.subplot(2, 2, 4, projection='polar')
    colors = plt.cm.tab10(range(len(df)))

    for idx, (_, row) in enumerate(df.iterrows()):
        # 计算归一化值 (越小越好: 显存、时间；越大越好: mAP)
        mem_norm = 1 - (row["显存占用 (GiB)"] / df["显存占用 (GiB)"].max())
        time_norm = 1 - (row["迭代时间 (s)"] / df["迭代时间 (s)"].max())
        map_norm = row["mAP@0.5 (%)"] / 100

        values = [mem_norm, time_norm, map_norm]
        values += values[:1]

        ax4.plot(angles, values, 'o-', linewidth=2, label=row["实验"], color=colors[idx])
        ax4.fill(angles, values, alpha=0.15, color=colors[idx])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('综合性能对比', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax4.grid(True)

    plt.tight_layout()

    # 保存图表
    plot_file = output_path / "comparison_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {plot_file}")

    # 保存为PDF
    pdf_file = output_path / "comparison_plot.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"图表已保存至: {pdf_file}")

    plt.close()


def save_comparison_report(df, results, output_path):
    """
    保存对比报告

    Args:
        df: pandas DataFrame
        results: 原始结果字典
        output_path: 输出路径
    """
    output_path = Path(output_path)

    # 创建报告文件
    report_file = output_path / "comparison_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv5训练优化实验对比报告\n")
        f.write("=" * 80 + "\n\n")

        # 表格
        f.write("实验结果对比:\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "=" * 80 + "\n\n")

        # 详细分析
        f.write("详细分析:\n")
        f.write("-" * 80 + "\n")

        for exp_name, exp_results in results.items():
            f.write(f"\n{exp_name}:\n")
            for key, value in exp_results.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"报告已保存至: {report_file}")

    # 同时保存为JSON
    json_file = output_path / "comparison_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"JSON结果已保存至: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="比较YOLOv5训练优化实验结果")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="实验输出目录")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="生成对比图表")

    args = parser.parse_args()

    # 加载结果
    print("加载实验结果...")
    results = load_results(args.output_dir)

    if not results:
        print("错误: 未找到实验结果")
        print(f"请确保 {args.output_dir} 目录包含实验结果文件")
        return

    print(f"找到 {len(results)} 个实验结果\n")

    # 创建对比表格
    df = create_comparison_table(results)

    # 打印对比
    print_comparison(df)

    # 保存报告
    save_comparison_report(df, results, args.output_dir)

    # 绘制图表
    if args.plot:
        try:
            plot_comparison(df, args.output_dir)
        except Exception as e:
            print(f"警告: 绘图失败: {e}")
            print("请确保已安装 matplotlib 和 seaborn")


if __name__ == "__main__":
    main()
