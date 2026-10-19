#!/bin/bash
# 100样本全量重跑: 5方法 + 4消融(v11全英文prompt), 串行执行
set -e
cd /root/autodl-tmp/vision_qa_platform
PY=/root/miniconda3/envs/vqa_env/bin/python
RES=results/accuracy

run_task() {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] START $name" >> $RES/queue_all100.log
    $PY "$@" >> $RES/queue_all100.log 2>&1
    echo "[$(date +%H:%M:%S)] DONE $name" >> $RES/queue_all100.log
}

# 5 方法同架构对比 (7B 双模型)
run_task standard_100 run_standard_qwen_v2.py --max_samples 100 --output_file "$RES/standard_100.json"
run_task cot_100 run_cot_qwen_v2.py --max_samples 100 --output_file "$RES/cot_100.json"
run_task tot_100 run_tot_on_gqa_v2.py --max_samples 100 --output_file "$RES/tot_100.json"
run_task got_100 run_got_on_gqa_v2.py --max_samples 100 --output_file "$RES/got_100.json"

# 四模块消融 (v11 全英文 prompt)
run_task modular_full_100_v11 run_modular_reasoning_on_gqa_v2.py --max_samples 100 --ablation full --output_file "$RES/modular_full_100_v11.json"
run_task abl100_v11_nodecompose run_modular_reasoning_on_gqa_v2.py --max_samples 100 --ablation no_decompose --output_file "$RES/modular_abl100_v11_nodecompose.json"
run_task abl100_v11_nogenerate run_modular_reasoning_on_gqa_v2.py --max_samples 100 --ablation no_generate --output_file "$RES/modular_abl100_v11_nogenerate.json"
run_task abl100_v11_noevaluate run_modular_reasoning_on_gqa_v2.py --max_samples 100 --ablation no_evaluate --output_file "$RES/modular_abl100_v11_noevaluate.json"
run_task abl100_v11_noaggregate run_modular_reasoning_on_gqa_v2.py --max_samples 100 --ablation no_aggregate --output_file "$RES/modular_abl100_v11_noaggregate.json"

echo "[$(date +%H:%M:%S)] ALL_DONE" >> $RES/queue_all100.log
