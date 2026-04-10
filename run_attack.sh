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
MODEL_NAME="ViT-L-14"
DATASET_TYPE="csv"
CSV_SEP=","
CSV_IMG_KEY="filepath"
CSV_CAPTION_KEY="title"
CSV_URL_KEY="url"
# run_attack.sh
PRETRAINED_TAG_OR_PATH="/home/sunyw/cliperase_ddp/open_clip/src/logs/cu_standing_man_lbase_1.0_lpos1.0_lredir1.0_lr1e-5_openai_f_ori2_fe0_erase/checkpoints/epoch_5.pt"
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
CSV_DIR="/home/sunyw/cliperase_ddp/datasets/vaw/vaw_dataset/data/man_standing/mia_csvs"

TRAIN_FORGET_CSV="$CSV_DIR/train_forget.csv"
TRAIN_RETAIN_CSV="$CSV_DIR/train_retain.csv"
TEST_FORGET_CSV="$CSV_DIR/test_forget.csv"
TEST_RETAIN_CSV="$CSV_DIR/test_retain.csv"

# -------- 3) 样本计数（请和CSV真实行数一致或接近） --------
# [OK] train_forget: 422
# [OK] train_retain: 4220
# [OK] test_forget:  7100
# [OK] test_retain:  7100
TRAIN_FORGET_NUM=422
TRAIN_RETAIN_NUM=4220
TEST_FORGET_NUM=7100
TEST_RETAIN_NUM=7100

# -------- 4) 参数映射说明 --------
# - member侧：forget
# - non-member侧：retain
# - *_1/2/3 接口要求三套，这里先复用同一路径

python3 main.py \
  --model "$MODEL_NAME" \
  --pretrained "$PRETRAINED_TAG_OR_PATH" \
  --dataset-type "$DATASET_TYPE" \
  --csv-separator "$CSV_SEP" \
  --csv-img-key "$CSV_IMG_KEY" \
  --csv-caption-key "$CSV_CAPTION_KEY" \
  --csv-url-key "$CSV_URL_KEY" \
  --mia-view "image" \
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
