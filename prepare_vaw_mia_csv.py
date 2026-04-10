#!/usr/bin/env python3
"""
Prepare CSV files for CLIP-MIA from custom VAW-style JSONL data.

Supported input formats:
1) Train JSONL (flat record), e.g.
   {
     "instance_id": "2402804010",
     "positive_caption": "A photo of happy smiling standing man.",
     "crop_path": "/.../2402804010.jpg",
     "flag": "forget"
   }
2) Test pair JSONL, e.g.
   {
     "sample": {... forget sample ...},
     "same_unit_sample": {... retain sample ...}
   }

Outputs (CSV with columns: filepath,title,url):
- train_forget.csv
- train_retain.csv
- test_forget.csv
- test_retain.csv
"""

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_filepath(sample, image_root: Path | None):
    if sample.get("crop_path"):
        return str(sample["crop_path"])
    instance_id = sample.get("instance_id")
    image_id = sample.get("image_id")
    if image_root is None:
        raise ValueError(f"Missing crop_path and image_root for sample: {sample}")
    if instance_id:
        return str(image_root / f"{instance_id}.jpg")
    if image_id:
        return str(image_root / f"{image_id}.jpg")
    raise ValueError(f"Cannot infer image path for sample: {sample}")


def to_row(sample, image_root: Path | None):
    caption = sample.get("positive_caption") or sample.get("caption")
    if not caption:
        raise ValueError(f"Missing caption text in sample: {sample}")
    filepath = build_filepath(sample, image_root)
    url = str(sample.get("instance_id") or sample.get("image_id") or filepath)
    return {"filepath": filepath, "title": caption, "url": url}


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "title", "url"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, default=None)
    parser.add_argument("--test-image-root", type=Path, default=None)
    parser.add_argument("--forget-flag", type=str, default="forget")
    args = parser.parse_args()

    train = read_jsonl(args.train_jsonl)
    test = read_jsonl(args.test_jsonl)

    train_forget, train_retain = [], []
    for item in train:
        row = to_row(item, args.train_image_root)
        if str(item.get("flag", "")).lower() == args.forget_flag.lower():
            train_forget.append(row)
        else:
            train_retain.append(row)

    test_forget, test_retain = [], []
    for item in test:
        if "sample" in item:
            test_forget.append(to_row(item["sample"], args.test_image_root))
        if "same_unit_sample" in item:
            test_retain.append(to_row(item["same_unit_sample"], args.test_image_root))

    out = args.output_dir
    write_csv(out / "train_forget.csv", train_forget)
    write_csv(out / "train_retain.csv", train_retain)
    write_csv(out / "test_forget.csv", test_forget)
    write_csv(out / "test_retain.csv", test_retain)

    print(f"[OK] train_forget: {len(train_forget)}")
    print(f"[OK] train_retain: {len(train_retain)}")
    print(f"[OK] test_forget:  {len(test_forget)}")
    print(f"[OK] test_retain:  {len(test_retain)}")
    print(f"[OK] csv dir: {out}")


if __name__ == "__main__":
    main()
