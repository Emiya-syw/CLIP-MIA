#!/usr/bin/env bash
set -euo pipefail

# 最简单运行脚本：用于启动 main.py
#
# 用法示例：
#   bash run_attack.sh ViT-B-32
#   bash run_attack.sh ViT-B-32 --batch-size 256 --nt-length 1000
#
# 参数说明：
#   参数1: MODEL_NAME（必填）
#     - 作用：指定要攻击的 CLIP 模型名称。
#     - 常见值：ViT-B-32 / ViT-B-16 / ViT-L-14 / RN50
#
#   参数2及之后: 透传给 main.py 的额外参数（可选）
#     - 作用：覆盖 main.py / params.py 中的默认超参数与数据路径。
#     - 示例：
#         --batch-size 256        # 每批样本数
#         --nt-length 1000        # 非成员样本采样数
#         --t-length 1000         # 伪成员样本采样数
#         --eval-length 2000      # 评估样本数
#         --train-data <path>     # 训练数据路径
#         --val-data <path>       # 验证数据路径

if [[ $# -lt 1 ]]; then
  echo "用法: bash run_attack.sh <MODEL_NAME> [main.py 的其他参数...]" >&2
  exit 1
fi

MODEL_NAME="$1"
shift

# 执行主程序：
# --model 使用上面传入的 MODEL_NAME
# "$@" 会把其余参数原样传给 main.py
python3 main.py --model "$MODEL_NAME" "$@"
