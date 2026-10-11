# 3) Interpretability（可解释性）

这项本质是“过程行为诊断”，不是单纯比谁分高。  
基于 `interpretability_metrics_complex50.csv` 的统计结果如下：

- 样本数：`298`（有效复杂样本）
- `correct_exact` 占比：`36.24%`（108 / 298）

## 正确样本 vs 错误样本（过程特征对比）

| 指标 | 正确样本均值 | 错误样本均值 | 现象解读 |
|------------------------|------------:|------------:|------------------------------|
| `depth_subquestions`   |        1.70 |        1.95 | 正确样本更浅，过深可能引入噪声 |
| `selected_avg_token_len` |      3.25 |        2.57 | 适中长度候选更有利于最终正确 |
| `avg_branch_factor`    |        2.31 |        2.41 | 错误样本分支略多，搜索更发散 |
| `avg_candidate_score`  |        0.52 |        0.55 | 高分候选不一定更正确，评分待校准 |
| `selected_empty_ratio` |        0.11 |        0.09 | 正确样本也存在空选现象，需进一步分析 |

## 关键解读

- “更深”不一定“更对”，可能出现过度分解与路径噪声累积；
- 候选文本并非越短越好，适度信息量更有利于命中正确答案；
- 单看候选分数存在偏差，后续可加强评估器与最终正确性的对齐。

---

Source:
- `results/interpretability/interpretability_metrics_complex50.csv`
