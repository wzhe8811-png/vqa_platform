# 基于推理时拓展的视觉问答平台

本科毕业设计《基于推理时拓展的视觉问答平台的设计与实现》的完整代码仓库。

本项目实现了一个**双模型协同**的视觉问答平台：LLaVA-1.5-7B 负责视觉感知，Qwen2-7B-Instruct 负责文本推理，并在推理阶段（inference time）引入五种逐步增强的推理拓展策略。平台提供 Flask Web 可视化界面，支持在线问答、五方法对比、思维树轨迹可视化、中英双语提示词，可实时观察推理过程的每一步。

## 核心方法

| 方法 | 思路 |
|------|------|
| **Standard** | LLaVA 视觉描述 + Qwen2 直接生成答案（基准） |
| **CoT** 思维链 | 引导模型输出逐步推理过程再作答 |
| **ToT** 树状思维 | 生成多个候选思维，评分后回溯搜索最优路径 |
| **GoT** 图状思维 | 在 ToT 基础上允许思维节点合并与聚合，形成有向无环图 |
| **Modular** 模块化推理 | 四模块流水线：问题分解器 → 思维生成器 → 状态评估器 → 搜索控制器 |

## 实验结果（GQA complex 100 样本，7B 双模型公平对比）

### 主实验：五种推理策略

| 方法 | 准确率 (%) |
|------|-----------:|
| Standard | 58.0 |
| CoT | 53.0 |
| ToT | 57.0 |
| GoT | **59.0** |
| Modular（四模块） | 58.0 |

### 消融实验：逐模块移除

| 配置 | 准确率 (%) |
|------|-----------:|
| Modular 完整版 | 58.0 |
| − 问题分解器 | 57.0 |
| − 思维生成器（视觉答案兜底） | 60.0 |
| − 状态评估器 | 57.0 |
| − 搜索控制器（直接聚合） | 55.0 |

### 鲁棒性实验：3 次重复运行

| 运行 | 准确率 (%) |
|------|-----------:|
| Run 1 | 60.0 |
| Run 2 | 58.0 |
| Run 3 | 58.0 |
| **均值** | **58.7 ± 1.2** |

完整结果与逐题 trace 见 `results/accuracy/`（`*_100.json` 为 100 样本主实验，`modular_abl100_*` 为消融，`modular_7b_*` 为鲁棒性重复运行）。更早的 1000 样本实验数据见 `results/accuracy/*_complex1000.json`。

## Web 可视化平台

`web_app.py` + `web_static/` 提供单页 Flask 应用（默认 `0.0.0.0:8080`），包含五个标签页：

| 标签页 | 功能 |
|--------|------|
| 在线问答 | 上传图片 + 输入问题，选择推理方法，实时查看答案与推理过程 |
| 五方法对比 | 同一问题跑 5 种方法并排对比 |
| 思维树可视化 | Modular 方法的完整推理树：问题 → 子问题（含视觉答案）→ 候选思维（含评分）→ 最终答案 |
| 实验数据 | 浏览 100 样本实验的逐题结果与统计 |
| 中英切换 | 全部提示词支持中英双语（`/api/translate`） |

## 目录结构

```
vqa_platform/
├── web_app.py                        # Flask 可视化平台（端口 8080）
├── web_static/                       # 前端页面（思维树 ECharts 可视化）
│   ├── index.html
│   └── echarts.min.js
├── reasoning_modules.py              # Modular 四模块推理引擎（v3：视觉答案注入）
├── tot_got_integration.py            # ToT/GoT 集成（视觉描述 + 推理）
├── vqa_eval_utils.py                 # 评测工具（精确/软匹配，双模型加载）
├── run_standard_qwen_v2.py           # Standard 评测（v2 脚本，100 样本体系）
├── run_cot_qwen_v2.py                # CoT 评测
├── run_tot_on_gqa_v2.py              # ToT 评测
├── run_got_on_gqa_v2.py              # GoT 评测
├── run_modular_reasoning_on_gqa_v2.py # Modular 评测（含消融模式）
├── run_all100.sh                     # 100 样本 5 方法 + 4 消融一键串行
├── run_ablation100.sh                # 消融实验一键运行
├── run_ablation_robust.sh            # 鲁棒性 3 次重复实验
├── stat_final100.py / stat_all100.py # 实验结果统计
├── patch_english_v11.py              # 全英文提示词补丁（v11）
├── results/                          # 论文实验数据与图表
│   └── accuracy/                     # 100 样本新体系 + 1000 样本旧体系
├── run_standard_on_gqa.py            # 旧版评测脚本（1000 样本体系，保留）
├── prepare_gqa_data.py               # GQA 数据预处理
├── agent_workspace/                  # 早期 Web 平台（三个服务，已被 web_app.py 取代）
├── deploy/
│   └── vqa_nginx.conf                # nginx 反向代理配置
├── LLaVA/                            # LLaVA-1.5 官方源码（vendor）
├── graph-of-thoughts/                # GoT 参考实现（vendor）
└── tree-of-thought-llm-master/       # ToT 参考实现（vendor）
```

## 环境要求

- Linux（推荐 AutoDL 等 GPU 云服务器，单卡 24GB 显存即可）
- Python 3.10，conda 环境
- 依赖：`transformers>=4.40`（含 AutoProcessor）、`torch`、`accelerate`、`flask`、`pillow`

```bash
conda create -n vqa_env python=3.10 -y
conda activate vqa_env
pip install torch transformers accelerate flask pillow
```

## 模型下载

两个模型均从 HuggingFace 下载（国内服务器可用 hf-mirror.com 镜像）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1   # hf-mirror 不支持 Xet 协议时必须设置

huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir llava-1.5-7b-hf
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir text_model
```

## 运行方式

**启动 Web 平台：**

```bash
python web_app.py   # 监听 0.0.0.0:8080
```

访问 `http://<服务器IP>:8080` 即可使用。上传图片 → 输入问题 → 选择推理方法（Standard/CoT/ToT/GoT/Modular）→ 查看答案与完整推理过程。

**批量评测（100 样本主实验 + 消融）：**

```bash
bash run_all100.sh        # 串行运行 5 方法 + 4 消融，结果写入 results/accuracy/
python stat_final100.py   # 统计准确率
```

**鲁棒性重复实验：**

```bash
bash run_ablation_robust.sh   # Modular 方法 3 次重复运行
```

## 数据

GQA 数据集需自行下载并预处理：

```bash
python prepare_gqa_data.py
```

评测样本为 GQA 中的复杂问题子集（多跳推理问题），100 样本体系与 1000 样本体系的划分见脚本内说明。
