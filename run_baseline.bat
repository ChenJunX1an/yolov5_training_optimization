@echo off
REM 运行基准组实验

python train.py ^
    --experiment baseline ^
    --model_type %1 ^
    --config configs\baseline_config.json ^
    --output_dir ./outputs/baseline ^
    --epochs 50 ^
    --batch_size 8

pause
