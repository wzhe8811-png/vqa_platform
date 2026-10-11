# 6.3 功能与可解释性测试

## 6.3.1 功能测试

为验证平台功能完整性，本文围绕“输入-推理-输出-展示”链路进行测试，重点检查系统是否能够稳定完成多模态问答、结构化推理和结果落盘。

测试内容与结果如下：

| 功能模块 | 测试方法 | 结果 |
|---|---|---|
| 多模态输入与语义理解 | 输入图像与问题，检查是否生成视觉描述并进入推理流程 | 通过 |
| 多方法推理执行 | 分别运行 Standard、CoT、ToT、GoT、Modular 脚本 | 通过 |
| 鲁棒性流程 | 对噪声等级 0.00/0.03/0.06/0.09/0.12 批量执行并汇总 | 通过 |
| 结果统计与可视化 | 生成对比表、鲁棒性曲线、可解释性案例 | 通过 |
| 消融流程 | 运行 full/no_decomposer/no_evaluator/no_searcher 并输出表格 | 通过 |

对应产物包括：

- 准确率对比：`results/baseline_comparison_complex20.csv`
- 鲁棒性汇总：`results/robustness_summary_complex50.csv`
- 鲁棒性图：`results/robustness_curve_exact_complex50.png`、`results/robustness_curve_soft_complex50.png`
- 可解释性案例：`results/figure4_interpretability_cases_complex50.md`
- 消融结果：`results/ablation_table_complex50_full.csv`

实验表明，系统在复杂样本子集上可稳定完成端到端推理与评测流程，且各模块输出格式一致，能够支持后续论文分析与复现实验。

## 6.3.2 可解释性测试

本节从“过程可见性”和“人工可评性”两个维度进行测试。平台在 `Modular(4-Module)` 推理过程中保留了子问题拆解、候选路径、评分与选择结果，可用于构建可解释分析。

### （1）过程可见性验证

基于 `results/modular_gaussian_0.00.json` 自动提取推理轨迹并构建案例图，输出文件为 `results/figure4_interpretability_cases_complex50.md`。  
该文件提供了典型样本的 Mermaid 思维图，能够显示：

- 子问题节点（SubQ）
- 候选答案及评分
- 被选分支（标记）
- 最终预测与真实答案对照

这说明系统已具备“推理路径可追溯”能力，而非仅输出黑盒答案。

### （2）可解释性定量统计

由 `results/interpretability_metrics_complex50.csv` 统计得到（complex50, clean, 共 31 题）：

- 平均推理深度：`2.00`
- 平均分支因子：`2.20`
- 候选平均评分：`0.563`
- 选中答案空值率：`0.075`
- 选中答案平均 token 长度：`28.24`

进一步比较正确与错误样本，正确样本平均深度为 `1.83`，错误样本平均深度为 `2.11`。该结果说明当前实现中，推理链条更深并不必然带来更高正确率，提示后续应进一步优化路径筛选与剪枝策略。

### （3）人工评估可执行性

系统已生成人工评分模板 `results/interpretability_human_scoring_template.csv`，包含逻辑一致性、证据相关性、冗余度、总体可解释性等维度，支持多人打分与后续统计分析。这为“定性 + 定量”联合评价提供了可执行方案。

## 6.3.3 本章小结

功能测试表明，平台已完成从输入、推理到评测与可视化的完整闭环；可解释性测试表明，系统能够输出可追踪的中间推理轨迹，并可通过人工评分模板开展主观评估。整体上，平台不仅具备问答性能测试能力，也具备面向毕业论文要求的可解释性验证能力。
