@echo off
REM YOLOv5 Training Optimization - 运行所有实验 (Windows)

echo ========================================
echo YOLOv5 Training Optimization Experiments
echo ========================================
echo.

REM 检查CUDA
echo 检查CUDA环境...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

REM 输出目录
set OUTPUT_DIR=./outputs
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 模型类型 (默认YOLOv5s)
set MODEL_TYPE=%1
if "%MODEL_TYPE%"=="" set MODEL_TYPE=s

echo.
echo 模型类型: YOLOv5%MODEL_TYPE%
echo 输出目录: %OUTPUT_DIR%
echo.

REM 运行各个实验
for %%e in (
    "baseline:baseline_config.json"
    "mixed_precision:mixed_precision_config.json"
    "gradient_checkpoint:gradient_checkpoint_config.json"
    "mixed_optimization:mixed_optimization_config.json"
    "zero3:zero3_config.json"
    "ablation:ablation_config.json"
) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%e") do (
        set exp_name=%%a
        set config_file=%%b
    )

    echo ========================================
    echo 运行实验: !exp_name!
    echo 配置文件: !config_file!
    echo ========================================

    REM 创建实验子目录
    set EXP_OUTPUT_DIR=%OUTPUT_DIR%\!exp_name!
    if not exist !EXP_OUTPUT_DIR! mkdir !EXP_OUTPUT_DIR!

    REM 运行训练
    python train.py ^
        --experiment !exp_name! ^
        --model_type %MODEL_TYPE% ^
        --config configs\!config_file! ^
        --output_dir !EXP_OUTPUT_DIR! ^
        --epochs 50 ^
        --batch_size 8 ^
        2>&1 | tee !EXP_OUTPUT_DIR!\training.log

    if !errorlevel! equ 0 (
        echo.
        echo [成功] 实验 !exp_name! 完成
        echo.
    ) else (
        echo.
        echo [失败] 实验 !exp_name! 失败
        echo.
    )
)

echo ========================================
echo 所有实验完成！
echo ========================================
echo.

REM 生成对比报告
echo 生成对比报告...
python scripts\compare_experiments.py --output_dir %OUTPUT_DIR%

pause
