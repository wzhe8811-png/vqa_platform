#!/bin/bash
# 消融实验(4种缺一模块) + 鲁棒性(完整版×2次), 各50样本, 串行执行
set -e
cd /root/autodl-tmp/vision_qa_platform
PY=/root/miniconda3/envs/vqa_env/bin/python
RUN=run_modular_reasoning_on_gqa_v2.py
RES=results/accuracy

run_task() {
    local name="$1"; local ablation="$2"
    echo "[$(date +%H:%M:%S)] START $name" >> $RES/queue_ablation.log
    $PY $RUN --max_samples 50 --ablation "$ablation" --output_file "$RES/$name.json" >> $RES/queue_ablation.log 2>&1
    echo "[$(date +%H:%M:%S)] DONE $name" >> $RES/queue_ablation.log
}

# 四模块消融: 各缺一个模块
run_task modular_ablation_nodecompose_test50 no_decompose
run_task modular_ablation_nogenerate_test50 no_generate
run_task modular_ablation_noevaluate_test50 no_evaluate
run_task modular_ablation_noaggregate_test50 no_aggregate

# 鲁棒性: 完整版再跑2次 (第1次 = 已有 modular_7b_eng_test50, 60%)
run_task modular_7b_robust_run2 full
run_task modular_7b_robust_run3 full

echo "[$(date +%H:%M:%S)] ALL_DONE" >> $RES/queue_ablation.log
