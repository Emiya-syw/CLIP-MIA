#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CLIP-MIA 一键运行脚本（CSV/自定义数据版）
# 直接执行：bash run_attack.sh
#
# 先用 prepare_vaw_mia_csv.py 生成4个CSV：
#   train_forget.csv / train_retain.csv / test_forget.csv / test_retain.csv
# ============================================================

# -------- 1) 基础参数 --------
MODEL_NAME="ViT-B-32"
DATASET_TYPE="csv"
CSV_SEP=","
CSV_IMG_KEY="filepath"
CSV_CAPTION_KEY="title"
CSV_URL_KEY="url"

BATCH_SIZE=128
NT_LENGTH=2000
T_LENGTH=2000
EVAL_LENGTH=1000
HYPER_LAMBDA=1.5

# -------- 2) CSV路径（改成你自己的） --------
# 目录示例：/path/to/mia_csvs/
#   ├── train_forget.csv
#   ├── train_retain.csv
#   ├── test_forget.csv
#   └── test_retain.csv
CSV_DIR="/path/to/mia_csvs"

TRAIN_FORGET_CSV="$CSV_DIR/train_forget.csv"
TRAIN_RETAIN_CSV="$CSV_DIR/train_retain.csv"
TEST_FORGET_CSV="$CSV_DIR/test_forget.csv"
TEST_RETAIN_CSV="$CSV_DIR/test_retain.csv"

# -------- 3) 样本计数（请和CSV真实行数一致或接近） --------
TRAIN_FORGET_NUM=5000
TRAIN_RETAIN_NUM=20000
TEST_FORGET_NUM=1000
TEST_RETAIN_NUM=1000

# -------- 4) 参数映射说明 --------
# - member侧：forget
# - non-member侧：retain
# - *_1/2/3 接口要求三套，这里先复用同一路径

python3 main.py \
  --model "$MODEL_NAME" \
  --dataset-type "$DATASET_TYPE" \
  --csv-separator "$CSV_SEP" \
  --csv-img-key "$CSV_IMG_KEY" \
  --csv-caption-key "$CSV_CAPTION_KEY" \
  --csv-url-key "$CSV_URL_KEY" \
  --batch-size "$BATCH_SIZE" \
  --nt-length "$NT_LENGTH" \
  --t-length "$T_LENGTH" \
  --eval-length "$EVAL_LENGTH" \
  --hyper-lambda "$HYPER_LAMBDA" \
  --train-data "$TRAIN_FORGET_CSV" \
  --val-data "$TEST_RETAIN_CSV" \
  --train-num-samples "$TRAIN_FORGET_NUM" \
  --val-num-samples "$TEST_RETAIN_NUM" \
  --val-data-nontrain-1 "$TEST_RETAIN_CSV" \
  --val-data-nontrain-2 "$TEST_RETAIN_CSV" \
  --val-data-nontrain-3 "$TEST_RETAIN_CSV" \
  --val-num-samples-nontrain-1 "$TEST_RETAIN_NUM" \
  --val-num-samples-nontrain-2 "$TEST_RETAIN_NUM" \
  --val-num-samples-nontrain-3 "$TEST_RETAIN_NUM" \
  --val-data-train-1 "$TRAIN_FORGET_CSV" \
  --val-data-train-2 "$TRAIN_FORGET_CSV" \
  --val-data-train-3 "$TRAIN_FORGET_CSV" \
  --val-num-samples-train-1 "$TRAIN_FORGET_NUM" \
  --val-num-samples-train-2 "$TRAIN_FORGET_NUM" \
  --val-num-samples-train-3 "$TRAIN_FORGET_NUM" \
  --train-data-eval "$TEST_FORGET_CSV" \
  --train-num-samples-eval "$TEST_FORGET_NUM" \
  --val-data-eval-1 "$TEST_RETAIN_CSV" \
  --val-data-eval-2 "$TEST_RETAIN_CSV" \
  --val-data-eval-3 "$TEST_RETAIN_CSV" \
  --val-num-samples-eval-1 "$TEST_RETAIN_NUM" \
  --val-num-samples-eval-2 "$TEST_RETAIN_NUM" \
  --val-num-samples-eval-3 "$TEST_RETAIN_NUM"
