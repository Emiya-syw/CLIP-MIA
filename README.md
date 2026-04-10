# CLIPMIA

This is an official repository for Practical Membership Inference Attacks Against Large-Scale Multi-Modal Models: A Pilot Study (ICCV2023).

&nbsp;

In this work, we take a first step towards developing practical MIAs against large-scale multi-modal models.

We introduce a simple baseline strategy by thresholding the cosine similarity between text and image features of a target point and propose further enhancing the baseline by aggregating cosine similarity across transformations of the target. 

We also present a new weakly supervised attack method that leverages ground-truth non-members (e.g., obtained by using the publication date of a target model and the timestamps of the open data) to further enhance the attack.

&nbsp;

## Steps to Reproduce the Attacks

1. **Install Anaconda Environment**

   Create an anaconda environment to manage packages and dependencies [see environment.yml].

&nbsp;

2. **Download Dataset**

   Download datasets such as LAION, CC12M, and CC3M using the repository: [img2dataset](https://github.com/rom1504/img2dataset/tree/main).

&nbsp;

3. **Check Duplication**

   Utilize "Inspect_image_overlapping.ipynb" and "Inspect_text_overlapping.ipynb" to inspect and save the overlapping text, and URLs between datasets.

&nbsp;

4. **Hyperparameter Setting**

   - **Non-Train Datasets**

     - `args.val_data-nontrain-1`: Path for non-train dataset 1.
     - `args.val_num_samples-nontrain`: Number of samples in non-train dataset 1.
     - Repeat for other non-train datasets.

   &nbsp;

   - **Pseudo-Train Datasets**

     - `args.val_data-train`: Path for pseudo-train dataset 1.
     - `args.val_num_samples-nontrain`: Number of samples in pseudo-train dataset 1.
     - Repeat for other pseudo-train datasets.

   &nbsp;

   - **Evaluation Datasets**

     - `args.train_data-1`: Path for the primary training dataset.
     - `—train_num_samples-1`: Number of samples in the training dataset.
     - `args.val_data-1`: Path for the first non-training dataset.
     - `—val_num_samples-1`: Number of samples in the first non-training dataset.
     - Repeat for other evaluation datasets.

&nbsp;

5. **Execute the Attack**

   Run `main.py --model ViT-B-32` (with other arguments).


### Quick Run Script

现在支持你的自定义数据（forget / retain）流程：

1. 先把训练JSON/JSONL和测试pair JSON/JSONL转成CSV（`filepath,title,url`）：

```bash
python3 prepare_vaw_mia_csv.py \
  --train-file /home/sunyw/cliperase_ddp/datasets/vaw/vaw_dataset/data/man_standing/train_man_standing_ori.jsonl \
  --test-file /home/sunyw/cliperase_ddp/datasets/vaw/vaw_dataset/data/man_standing/CFR_irr.json \
  --output-dir /home/sunyw/cliperase_ddp/datasets/vaw/vaw_dataset/data/man_standing/mia_csvs \
  --test-image-root /home/sunyw/VL-CheckList-main/datasets/vg/VG_100K_2
```

2. 修改 `run_attack.sh` 里的 `CSV_DIR` 和样本数参数。

3. 直接运行：

```bash
bash run_attack.sh
```

说明：默认映射是 `forget -> member`，`retain -> non-member`；`CFR_irr.json` 这种数组JSON也可直接读取。

&nbsp;

For any questions, please contact [myeongseob@vt.edu](mailto:myeongseob@vt.edu).
