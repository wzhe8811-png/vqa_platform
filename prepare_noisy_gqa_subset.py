import argparse
import json
import os
from typing import List

import numpy as np
from PIL import Image


def parse_levels(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def add_gaussian_noise(image: Image.Image, sigma_ratio: float) -> Image.Image:
    if sigma_ratio <= 0:
        return image.copy()
    arr = np.array(image).astype(np.float32)
    sigma = 255.0 * sigma_ratio
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def main():
    parser = argparse.ArgumentParser(description="Prepare noisy GQA subset jsonl and images.")
    parser.add_argument(
        "--source_jsonl",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa/gqa_clean_test.jsonl",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/vision_qa_platform/data/processed_gqa_robustness",
    )
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--levels", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    levels = parse_levels(args.levels)

    os.makedirs(args.output_root, exist_ok=True)

    samples = []
    with open(args.source_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= args.max_samples:
                break

    if not samples:
        raise ValueError("No samples loaded from source jsonl.")

    for level in levels:
        level_tag = f"gaussian_{level:.2f}"
        level_dir = os.path.join(args.output_root, level_tag)
        image_out_dir = os.path.join(level_dir, "images")
        os.makedirs(image_out_dir, exist_ok=True)

        out_rows = []
        for i, row in enumerate(samples):
            src_img = row.get("full_image_path", "")
            if not src_img or not os.path.exists(src_img):
                continue

            img = Image.open(src_img).convert("RGB")
            noisy = add_gaussian_noise(img, level)

            out_name = f"{i:05d}.jpg"
            out_path = os.path.join(image_out_dir, out_name)
            noisy.save(out_path, quality=95)

            new_row = dict(row)
            new_row["full_image_path"] = out_path
            out_rows.append(new_row)

        out_jsonl = os.path.join(level_dir, "gqa_clean_test.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Prepared {len(out_rows)} samples at noise level {level:.2f}: {out_jsonl}")


if __name__ == "__main__":
    main()
