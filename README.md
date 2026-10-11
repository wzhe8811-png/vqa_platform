# 基于推理时拓展的视觉问答平台

本科毕业设计《基于推理时拓展的视觉问答平台的设计与实现》的完整代码仓库。

本项目实现了一个**双模型协同**的视觉问答平台：LLaVA-1.5-7B 负责视觉感知，Qwen2-1.5B-Instruct 负责文本推理，并在推理阶段（inference time）引入五种逐步增强的推理拓展策略，显著提升复杂视觉问答的准确率。平台包含离线轨迹浏览、在线问答、用户门户三个 Web 界面。

## 核心方法

| 方法 | 思路 |
|------|------|
| **Standard** | 单模型直接生成答案（基准） |
| **CoT** 思维链 | 引导模型输出逐步推理过程再作答 |
| **ToT** 树状思维 | 生成多个候选思维，评分后回溯搜索最优路径 |
| **GoT** 图状思维 | 在 ToT 基础上允许思维节点合并与聚合，形成有向无环图 |
| **Modular** 模块化推理 | 四模块流水线：问题分解器 → 思维生成器 → 状态评估器 → 搜索控制器 |

## 实验结果（GQA complex 1000 样本）

| 方法 | Exact Acc (%) | Soft Acc (%) |
|------|--------------:|-------------:|
| Standard | 17.80 | 18.40 |
| CoT | 25.90 | 26.80 |
| ToT | 29.40 | 33.10 |
| GoT | 29.40 | 34.00 |
| **Modular（4 模块）** | **33.40** | **36.70** |

完整结果与图表见 `results/accuracy/`。

## 目录结构

```
vqa_platform/
├── run_standard_on_gqa.py            # 标准方法评测
├── run_cot_on_gqa.py                 # 思维链评测
├── run_tot_on_gqa.py                 # 树状思维评测
├── run_got_on_gqa.py                 # 图状思维评测
├── run_modular_reasoning_on_gqa.py   # 模块化推理评测
├── reasoning_modules.py              # 推理方法通用模块
├── tot_got_integration.py            # ToT/GoT 集成（视觉描述 + 推理）
├── vqa_eval_utils.py                 # 评测工具（精确/软匹配）
├── prepare_gqa_data.py               # GQA 数据预处理
├── compare_gqa_baselines.py          # 基线对比与绘图
├── agent_workspace/                  # Web 平台（三个服务 + 共享配置）
│   ├── trace_webapp/                 # 离线轨迹浏览（端口 7860）
│   ├── chat_vqa_webapp/              # 在线问答（端口 7861）
│   ├── chat_vqa_portal/              # 用户门户 / SSO 登录（端口 7862）
│   ├── shared/                       # 跨服务共享 Session 配置
│   └── webapps_ctl.sh                # 一键启停脚本
├── deploy/
│   └── vqa_nginx.conf                # nginx 反向代理（6006 分流 / 6008 在线问答）
├── results/                          # 论文实验数据与图表
├── LLaVA/                            # LLaVA-1.5 官方源码（vendor）
├── graph-of-thoughts/                # GoT 参考实现（vendor）
└── tree-of-thought-llm-master/       # ToT 参考实现（vendor）
```

## 环境要求

- Linux（推荐 AutoDL 等 GPU 云服务器，单卡 24GB 显存即可）
- Python 3.10，conda 环境
- 依赖：`transformers>=4.40`（含 AutoProcessor）、`torch`、`accelerate`、`flask`、`bcrypt`、`pyarrow`、`pillow`

```bash
conda create -n vqa_env python=3.10 -y
conda activate vqa_env
pip install torch transformers accelerate flask bcrypt pyarrow pillow
```

## 模型下载

两个模型均从 HuggingFace 下载（国内服务器可用 hf-mirror.com 镜像）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1   # hf-mirror 不支持 Xet 协议时必须设置

huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir llava-1.5-7b-hf
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir qwen2_1_5b_instruct
```

模型路径在 `run_*.py` 与 `tot_got_integration.py` 的头部常量中配置。

## 数据准备

GQA 评测标注可从 HuggingFace 获取（lmms-lab-encoder/GQA 的 parquet），图片可用官方 GQA 数据集（约 20GB），或用 `data_sample/` 中的 10000 张样例快速验证（样例不随仓库分发，自行放入同名目录即可）。

```bash
python prepare_gqa_data.py   # 生成 data/processed_gqa/gqa_clean_test.jsonl
```

## 运行实验

```bash
python run_standard_on_gqa.py          # 标准方法
python run_cot_on_gqa.py               # 思维链
python run_tot_on_gqa.py               # 树状思维
python run_got_on_gqa.py               # 图状思维
python run_modular_reasoning_on_gqa.py # 模块化推理（4 模块）
```

结果统一写入 `results/accuracy/`，可用 `compare_gqa_baselines.py` 生成对比表与图表。

## Web 平台

三个 Flask 服务由 nginx 统一反代（配置见 `deploy/vqa_nginx.conf`）：

| 公网入口 | 路由 | 服务 |
|---------|------|------|
| 6006 | `/login` `/register` `/chat` `/api/auth/*` `/static/` | 用户门户（7862） |
| 6006 | 其余路径（轨迹浏览、案例图片等） | 离线轨迹浏览（7860） |
| 6008 | 全部 | 在线问答（7861） |

```bash
cd agent_workspace
./webapps_ctl.sh start       # 启动全部三个服务
./webapps_ctl.sh stop        # 停止
sudo cp deploy/vqa_nginx.conf /etc/nginx/conf.d/ && service nginx reload
```

初始管理员账号：`admin / admin123456`（可在启动前用 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 环境变量覆盖）。

## 已知兼容性说明

- 若使用 transformers 5.x：必须用 `AutoProcessor(text=..., images=...)` 生成输入（会自动展开 576 个 image token），不要手动拼 tokenizer + image_processor，否则会报 `Image features and image tokens do not match`。
- 生成答案时建议用 `generated_text.split("ASSISTANT:")[-1].strip()` 截取，避免长度切片截断首字符。

## License

本项目代码仅用于学术研究与毕业设计复现。第三方目录（LLaVA、graph-of-thoughts、tree-of-thought-llm-master）遵循各自原有许可。
