#!/bin/bash
# YOLOv5 Training Optimization - 运行所有实验

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}YOLOv5 Training Optimization Experiments${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查CUDA
echo -e "\n${YELLOW}检查CUDA环境...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 输出目录
OUTPUT_DIR="./outputs"
mkdir -p $OUTPUT_DIR

# 实验列表
experiments=(
    "baseline:baseline_config.json"
    "mixed_precision:mixed_precision_config.json"
    "gradient_checkpoint:gradient_checkpoint_config.json"
    "mixed_optimization:mixed_optimization_config.json"
    "zero3:zero3_config.json"
    "ablation:ablation_config.json"
)

# 模型类型
MODEL_TYPE=${1:-"s"}  # 默认使用YOLOv5s

echo -e "\n${YELLOW}模型类型: YOLOv5${MODEL_TYPE}${NC}"
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}\n"

# 运行每个实验
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name config_file <<< "$exp"

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}运行实验: $exp_name${NC}"
    echo -e "${GREEN}配置文件: $config_file${NC}"
    echo -e "${GREEN}========================================${NC}"

    # 创建实验子目录
    EXP_OUTPUT_DIR="$OUTPUT_DIR/$exp_name"
    mkdir -p $EXP_OUTPUT_DIR

    # 运行训练
    python train.py \
        --experiment $exp_name \
        --model_type $MODEL_TYPE \
        --config configs/$config_file \
        --output_dir $EXP_OUTPUT_DIR \
        --epochs 50 \
        --batch_size 8 \
        2>&1 | tee $EXP_OUTPUT_DIR/training.log

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 实验 $exp_name 完成${NC}\n"
    else
        echo -e "${RED}✗ 实验 $exp_name 失败${NC}\n"
    fi
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"

# 生成对比报告
echo -e "\n${YELLOW}生成对比报告...${NC}"
python scripts/compare_experiments.py --output_dir $OUTPUT_DIR
