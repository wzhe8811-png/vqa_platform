#!/bin/bash
# 100样本消融实验: full基准 + 4种缺一模块, 串行执行
set -e
cd /root/autodl-tmp/vision_qa_platform
PY=/root/miniconda3/envs/vqa_env/bin/python
RUN=run_modular_reasoning_on_gqa_v2.py
RES=results/accuracy

run_task() {
    local name="$1"; local ablation="$2"
    echo "[$(date +%H:%M:%S)] START $name" >> $RES/queue_ablation100.log
    $PY $RUN --max_samples 100 --ablation "$ablation" --output_file "$RES/$name.json" >> $RES/queue_ablation100.log 2>&1
    echo "[$(date +%H:%M:%S)] DONE $name" >> $RES/queue_ablation100.log
}

run_task modular_full_100 full
run_task modular_abl100_nodecompose no_decompose
run_task modular_abl100_nogenerate no_generate
run_task modular_abl100_noevaluate no_evaluate
run_task modular_abl100_noaggregate no_aggregate

echo "[$(date +%H:%M:%S)] ALL_DONE" >> $RES/queue_ablation100.log
