# Vision QA Runtime Package

This package is prepared for migration to another Linux machine.

## What this package includes

- Core project code under `vision_qa_platform/`
- Runtime models:
  - `vision_qa_platform/llava-1.5-7b-hf/`
  - `vision_qa_platform/text_model/qwen2_1_5b_instruct/`
- A lightweight 10-sample test dataset:
  - `vision_qa_platform/data/processed_gqa_robustness_1000/gaussian_0.00_test10/`
  - `gqa_clean_test.jsonl` + `images/00000.jpg` to `images/00009.jpg`

## Main method scripts (project root)

- `run_standard_on_gqa.py`
- `run_cot_on_gqa.py`
- `run_tot_on_gqa.py`
- `run_got_on_gqa.py`
- `run_modular_reasoning_on_gqa.py`

## Data and path notes

- This runtime test set is portable (no hardcoded absolute image paths in jsonl).
- Use this input directory for quick verification:
  - `data/processed_gqa_robustness_1000/gaussian_0.00_test10`

## Quick start

From project root:

```bash
python run_standard_on_gqa.py \
  --input_dir "data/processed_gqa_robustness_1000/gaussian_0.00_test10" \
  --max_samples 10 \
  --output_file "results/standard_test10.json"
```

```bash
python run_cot_on_gqa.py \
  --input_dir "data/processed_gqa_robustness_1000/gaussian_0.00_test10" \
  --max_samples 10 \
  --output_file "results/cot_test10.json"
```

```bash
python run_tot_on_gqa.py \
  --input_dir "data/processed_gqa_robustness_1000/gaussian_0.00_test10" \
  --max_samples 10 \
  --output_file "results/tot_test10.json"
```

```bash
python run_got_on_gqa.py \
  --input_dir "data/processed_gqa_robustness_1000/gaussian_0.00_test10" \
  --max_samples 10 \
  --output_file "results/got_test10.json"
```

```bash
python run_modular_reasoning_on_gqa.py \
  --input_dir "data/processed_gqa_robustness_1000/gaussian_0.00_test10" \
  --max_samples 10 \
  --output_file "results/modular_test10.json"
```

## Typical functionality

- `Standard`: direct VQA generation with LLaVA.
- `CoT`: direct VQA with chain-of-thought style instruction.
- `ToT`: LLaVA visual description + Qwen tree-of-thought candidate search.
- `GoT`: LLaVA visual description + Qwen graph-style sub-question reasoning.
- `Modular`: 4-module pipeline (decompose/generate/evaluate/search) with trace output.

## Important environment reminder

- Scripts currently use model paths rooted at this project directory.
- Keep the internal directory layout unchanged after extraction.
- Run scripts from inside `vision_qa_platform/`.
