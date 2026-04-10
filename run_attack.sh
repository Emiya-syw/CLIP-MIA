#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CLIP-MIA 一键运行脚本（参数写死版）
# 直接执行：
#   bash run_attack.sh
# ============================================================

# -------- 1) 必改/常改参数（按需修改） --------
MODEL_NAME="ViT-B-32"          # 目标CLIP模型: ViT-B-32 / ViT-B-16 / ViT-L-14 / RN50
BATCH_SIZE=256                  # batch size
NT_LENGTH=3000                  # non-member采样数量
T_LENGTH=5000                   # pseudo-member采样数量
EVAL_LENGTH=2000                # 评估样本数量
HYPER_LAMBDA=1.5                # 阈值系数 (mean + lambda * std)

# -------- 2) 数据路径参数（请改成你本机路径） --------
TRAIN_DATA="/path/to/laion/{00000..13000}.tar"
VAL_DATA="/path/to/cc3m/{00000..00099}.tar"

VAL_DATA_NONTRAIN_1="/path/to/cc3m/{00000..00099}.tar"
VAL_DATA_NONTRAIN_2="/path/to/cc12m/{00000..00399}.tar"
VAL_DATA_NONTRAIN_3="/path/to/mscoco/{00000..00019}.tar"

VAL_DATA_TRAIN_1="/path/to/cc3m/{00100..00199}.tar"
VAL_DATA_TRAIN_2="/path/to/cc12m/{00400..00799}.tar"
VAL_DATA_TRAIN_3="/path/to/mscoco/{00020..00039}.tar"

TRAIN_DATA_EVAL="/path/to/laion/{13000..26000}.tar"
VAL_DATA_EVAL_1="/path/to/cc3m/{00200..00299}.tar"
VAL_DATA_EVAL_2="/path/to/cc12m/{00800..01199}.tar"
VAL_DATA_EVAL_3="/path/to/mscoco/{00040..00059}.tar"

# -------- 3) 样本总数参数（按你的数据规模修改） --------
TRAIN_NUM_SAMPLES=130000000
VAL_NUM_SAMPLES=1000000
VAL_NUM_SAMPLES_NONTRAIN_1=1000000
VAL_NUM_SAMPLES_NONTRAIN_2=4000000
VAL_NUM_SAMPLES_NONTRAIN_3=200000
VAL_NUM_SAMPLES_TRAIN_1=1000000
VAL_NUM_SAMPLES_TRAIN_2=4000000
VAL_NUM_SAMPLES_TRAIN_3=200000
TRAIN_NUM_SAMPLES_EVAL=130000000
VAL_NUM_SAMPLES_EVAL_1=1000000
VAL_NUM_SAMPLES_EVAL_2=4000000
VAL_NUM_SAMPLES_EVAL_3=200000

# -------- 4) 执行 --------
python3 main.py \
  --model "$MODEL_NAME" \
  --batch-size "$BATCH_SIZE" \
  --nt-length "$NT_LENGTH" \
  --t-length "$T_LENGTH" \
  --eval-length "$EVAL_LENGTH" \
  --hyper-lambda "$HYPER_LAMBDA" \
  --train-data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --train-num-samples "$TRAIN_NUM_SAMPLES" \
  --val-num-samples "$VAL_NUM_SAMPLES" \
  --val-data-nontrain-1 "$VAL_DATA_NONTRAIN_1" \
  --val-data-nontrain-2 "$VAL_DATA_NONTRAIN_2" \
  --val-data-nontrain-3 "$VAL_DATA_NONTRAIN_3" \
  --val-num-samples-nontrain-1 "$VAL_NUM_SAMPLES_NONTRAIN_1" \
  --val-num-samples-nontrain-2 "$VAL_NUM_SAMPLES_NONTRAIN_2" \
  --val-num-samples-nontrain-3 "$VAL_NUM_SAMPLES_NONTRAIN_3" \
  --val-data-train-1 "$VAL_DATA_TRAIN_1" \
  --val-data-train-2 "$VAL_DATA_TRAIN_2" \
  --val-data-train-3 "$VAL_DATA_TRAIN_3" \
  --val-num-samples-train-1 "$VAL_NUM_SAMPLES_TRAIN_1" \
  --val-num-samples-train-2 "$VAL_NUM_SAMPLES_TRAIN_2" \
  --val-num-samples-train-3 "$VAL_NUM_SAMPLES_TRAIN_3" \
  --train-data-eval "$TRAIN_DATA_EVAL" \
  --train-num-samples-eval "$TRAIN_NUM_SAMPLES_EVAL" \
  --val-data-eval-1 "$VAL_DATA_EVAL_1" \
  --val-data-eval-2 "$VAL_DATA_EVAL_2" \
  --val-data-eval-3 "$VAL_DATA_EVAL_3" \
  --val-num-samples-eval-1 "$VAL_NUM_SAMPLES_EVAL_1" \
  --val-num-samples-eval-2 "$VAL_NUM_SAMPLES_EVAL_2" \
  --val-num-samples-eval-3 "$VAL_NUM_SAMPLES_EVAL_3"
