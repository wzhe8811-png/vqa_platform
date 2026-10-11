#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/vision_qa_platform"
PYTHON_DEFAULT="python"
PYTHON_LLAVA="/root/miniconda3/envs/llava_env/bin/python"
ACC_MAX_SAMPLES="${ACC_MAX_SAMPLES:-1000}"
ROBUST_MAX_SAMPLES="${ROBUST_MAX_SAMPLES:-1000}"
ABLATION_MAX_SAMPLES="${ABLATION_MAX_SAMPLES:-1000}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0,0.03,0.06,0.09,0.12}"
ROBUST_DATA_ROOT="${ROBUST_DATA_ROOT:-$PROJECT_ROOT/data/processed_gqa_robustness_${ROBUST_MAX_SAMPLES}}"
RESUME_MODE="${RESUME_MODE:-1}"

cd "$PROJECT_ROOT"

ACC_TAG="complex${ACC_MAX_SAMPLES}"
ROBUST_TAG="complex${ROBUST_MAX_SAMPLES}"
ABLATION_TAG="complex${ABLATION_MAX_SAMPLES}"
CKPT_DIR="$PROJECT_ROOT/results/.checkpoints"
mkdir -p "$CKPT_DIR"

ACC_CKPT="$CKPT_DIR/accuracy_${ACC_TAG}.done"
ROBUST_CKPT="$CKPT_DIR/robustness_${ROBUST_TAG}.done"
INTERP_CKPT="$CKPT_DIR/interpretability_${ROBUST_TAG}.done"
ABLATION_CKPT="$CKPT_DIR/ablation_${ABLATION_TAG}.done"

echo "Config:"
echo "  ACC_MAX_SAMPLES=$ACC_MAX_SAMPLES"
echo "  ROBUST_MAX_SAMPLES=$ROBUST_MAX_SAMPLES"
echo "  ABLATION_MAX_SAMPLES=$ABLATION_MAX_SAMPLES"
echo "  ROBUST_DATA_ROOT=$ROBUST_DATA_ROOT"
echo "  NOISE_LEVELS=$NOISE_LEVELS"
echo "  RESUME_MODE=$RESUME_MODE"

# Auto-prepare robustness dataset at requested scale if missing.
if [[ ! -s "$ROBUST_DATA_ROOT/gaussian_0.00/gqa_clean_test.jsonl" ]]; then
  echo "Preparing robustness subset at: $ROBUST_DATA_ROOT"
  "$PYTHON_DEFAULT" "$PROJECT_ROOT/prepare_noisy_gqa_subset.py" \
    --source_jsonl "$PROJECT_ROOT/data/processed_gqa/gqa_clean_test.jsonl" \
    --output_root "$ROBUST_DATA_ROOT" \
    --max_samples "$ROBUST_MAX_SAMPLES" \
    --levels "$NOISE_LEVELS"
fi

if [[ "$RESUME_MODE" == "1" && -e "$ACC_CKPT" ]]; then
  echo "== [1/4] Accuracy Experiment (${ACC_TAG}) [SKIP: checkpoint exists] =="
else
  echo "== [1/4] Accuracy Experiment (${ACC_TAG}) =="
  "$PYTHON_DEFAULT" run_standard_on_gqa.py --input_dir "$PROJECT_ROOT/data/processed_gqa" --max_samples "$ACC_MAX_SAMPLES" --complex_only --output_file "$PROJECT_ROOT/results/standard_results_${ACC_TAG}.json"
  "$PYTHON_DEFAULT" run_cot_on_gqa.py --input_dir "$PROJECT_ROOT/data/processed_gqa" --max_samples "$ACC_MAX_SAMPLES" --complex_only --output_file "$PROJECT_ROOT/results/cot_results_${ACC_TAG}.json"
  "$PYTHON_DEFAULT" run_tot_on_gqa.py --input_dir "$PROJECT_ROOT/data/processed_gqa" --max_samples "$ACC_MAX_SAMPLES" --complex_only --output_file "$PROJECT_ROOT/results/tot_results_${ACC_TAG}.json"
  "$PYTHON_DEFAULT" run_got_on_gqa.py --input_dir "$PROJECT_ROOT/data/processed_gqa" --max_samples "$ACC_MAX_SAMPLES" --complex_only --output_file "$PROJECT_ROOT/results/got_results_${ACC_TAG}.json"
  "$PYTHON_DEFAULT" run_modular_reasoning_on_gqa.py --input_dir "$PROJECT_ROOT/data/processed_gqa" --max_samples "$ACC_MAX_SAMPLES" --complex_only --output_file "$PROJECT_ROOT/results/modular_results_${ACC_TAG}.json"

  "$PYTHON_DEFAULT" compare_gqa_baselines.py \
    --standard_file "$PROJECT_ROOT/results/standard_results_${ACC_TAG}.json" \
    --cot_file "$PROJECT_ROOT/results/cot_results_${ACC_TAG}.json" \
    --tot_file "$PROJECT_ROOT/results/tot_results_${ACC_TAG}.json" \
    --got_file "$PROJECT_ROOT/results/got_results_${ACC_TAG}.json" \
    --modular_file "$PROJECT_ROOT/results/modular_results_${ACC_TAG}.json" \
    --output_md "$PROJECT_ROOT/results/baseline_comparison_${ACC_TAG}.md" \
    --output_csv "$PROJECT_ROOT/results/baseline_comparison_${ACC_TAG}.csv"

  "$PYTHON_DEFAULT" plot_baseline_comparison.py \
    --input_csv "$PROJECT_ROOT/results/baseline_comparison_${ACC_TAG}.csv" \
    --output_png "$PROJECT_ROOT/results/baseline_comparison_${ACC_TAG}.png"
  touch "$ACC_CKPT"
fi

if [[ "$RESUME_MODE" == "1" && -e "$ROBUST_CKPT" ]]; then
  echo "== [2/4] Robustness Experiment (${ROBUST_TAG}) [SKIP: checkpoint exists] =="
else
  echo "== [2/4] Robustness Experiment (${ROBUST_TAG}) =="
  "$PYTHON_DEFAULT" run_robustness_experiment.py \
    --levels "$NOISE_LEVELS" \
    --max_samples "$ROBUST_MAX_SAMPLES" \
    --complex_only \
    --data_root "$ROBUST_DATA_ROOT" \
    --summary_csv "$PROJECT_ROOT/results/robustness_summary_${ROBUST_TAG}.csv" \
    --python_exec "$PYTHON_LLAVA"

  "$PYTHON_DEFAULT" plot_robustness_curve.py \
    --input_csv "$PROJECT_ROOT/results/robustness_summary_${ROBUST_TAG}.csv" \
    --metric exact_acc \
    --output_png "$PROJECT_ROOT/results/robustness_curve_exact_${ROBUST_TAG}.png"

  "$PYTHON_DEFAULT" plot_robustness_curve.py \
    --input_csv "$PROJECT_ROOT/results/robustness_summary_${ROBUST_TAG}.csv" \
    --metric soft_acc \
    --output_png "$PROJECT_ROOT/results/robustness_curve_soft_${ROBUST_TAG}.png"
  touch "$ROBUST_CKPT"
fi

if [[ "$RESUME_MODE" == "1" && -e "$INTERP_CKPT" ]]; then
  echo "== [3/4] Interpretability Artifacts [SKIP: checkpoint exists] =="
else
  echo "== [3/4] Interpretability Artifacts =="
  "$PYTHON_DEFAULT" "$PROJECT_ROOT/agent_workspace/generate_interpretability_artifacts.py"
  touch "$INTERP_CKPT"
fi

if [[ "$RESUME_MODE" == "1" && -e "$ABLATION_CKPT" ]]; then
  echo "== [4/4] Ablation Study (${ABLATION_TAG} full) [SKIP: checkpoint exists] =="
else
  echo "== [4/4] Ablation Study (${ABLATION_TAG} full) =="
  "$PYTHON_LLAVA" "$PROJECT_ROOT/agent_workspace/run_ablation_study.py" \
    --input_dir "$ROBUST_DATA_ROOT/gaussian_0.00" \
    --max_samples "$ABLATION_MAX_SAMPLES" \
    --complex_only \
    --output_json "$PROJECT_ROOT/results/ablation_raw_${ABLATION_TAG}_full.json" \
    --output_csv "$PROJECT_ROOT/results/ablation_table_${ABLATION_TAG}_full.csv" \
    --output_md "$PROJECT_ROOT/results/ablation_table_${ABLATION_TAG}_full.md"
  touch "$ABLATION_CKPT"
fi

echo ""
echo "All experiments completed."
echo "Key outputs:"
echo "  - Accuracy Table:      $PROJECT_ROOT/results/baseline_comparison_${ACC_TAG}.csv"
echo "  - Robustness Summary:  $PROJECT_ROOT/results/robustness_summary_${ROBUST_TAG}.csv"
echo "  - Figure 4 Cases:      $PROJECT_ROOT/results/figure4_interpretability_cases_complex50.md"
echo "  - Ablation Table 3:    $PROJECT_ROOT/results/ablation_table_${ABLATION_TAG}_full.csv"
echo "  - Checkpoints:         $CKPT_DIR"
