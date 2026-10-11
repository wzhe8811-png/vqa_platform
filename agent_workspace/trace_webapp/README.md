# 推理过程可视化交互（Flask + D3.js）

## 1) 安装依赖

```bash
/root/miniconda3/envs/llava_env/bin/pip install flask
```

## 2) 启动服务

```bash
cd /root/autodl-tmp/vision_qa_platform/agent_workspace/trace_webapp
/root/miniconda3/envs/llava_env/bin/python app.py
```

默认地址：`http://127.0.0.1:7860`

## 3) 切换数据文件（可选）

默认读取：

`/root/autodl-tmp/vision_qa_platform/results/accuracy/modular_results_complex1000.json`

可通过环境变量覆盖：

```bash
TRACE_RESULTS_FILE="/root/autodl-tmp/vision_qa_platform/results/accuracy/modular_results_complex20.json" \
/root/miniconda3/envs/llava_env/bin/python app.py
```

## 4) 功能说明

- 左侧：案例列表，按 `question_id` 切换样本
- 中间：D3 动态思维图（支持拖拽）
- 右侧：点击节点后显示详细内容（子问题、候选 thought/answer、评分、是否被选中）
