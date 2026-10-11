from pathlib import Path

import pandas as pd


def md_table(pivot_df: pd.DataFrame) -> str:
    cols = ["noise_level"] + list(pivot_df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for idx, row in pivot_df.iterrows():
        vals = [f"{idx:.2f}"] + [f"{float(row[c]):.2f}" for c in pivot_df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    csv_path = Path("/root/autodl-tmp/vision_qa_platform/results/robustness_summary_complex50.csv")
    out_path = Path("/root/autodl-tmp/vision_qa_platform/results/robustness_report_complex50.md")

    df = pd.read_csv(csv_path)
    df["noise_level"] = df["noise_level"].astype(float)

    methods = sorted(df["method"].unique().tolist())
    levels = sorted(df["noise_level"].unique().tolist())

    exact = df.pivot(index="noise_level", columns="method", values="exact_acc").sort_index()
    soft = df.pivot(index="noise_level", columns="method", values="soft_acc").sort_index()

    summary = []
    for m in methods:
        e0 = float(exact.loc[levels[0], m])
        e1 = float(exact.loc[levels[-1], m])
        s0 = float(soft.loc[levels[0], m])
        s1 = float(soft.loc[levels[-1], m])
        e_mean = float(exact[m].mean())
        s_mean = float(soft[m].mean())
        e_drop = e0 - e1
        s_drop = s0 - s1
        summary.append(
            {
                "method": m,
                "exact@0.00": e0,
                "exact@max": e1,
                "exact_mean": e_mean,
                "exact_drop": e_drop,
                "soft@0.00": s0,
                "soft@max": s1,
                "soft_mean": s_mean,
                "soft_drop": s_drop,
            }
        )

    sum_df = pd.DataFrame(summary).sort_values(["soft_mean", "exact_mean"], ascending=False)

    rank_cols = [
        "method",
        "exact@0.00",
        "exact@max",
        "exact_mean",
        "exact_drop",
        "soft@0.00",
        "soft@max",
        "soft_mean",
        "soft_drop",
    ]
    rank_lines = []
    rank_lines.append("| " + " | ".join(rank_cols) + " |")
    rank_lines.append("| " + " | ".join(["---"] * len(rank_cols)) + " |")
    for _, r in sum_df.iterrows():
        rank_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["method"]),
                    f"{r['exact@0.00']:.2f}",
                    f"{r['exact@max']:.2f}",
                    f"{r['exact_mean']:.2f}",
                    f"{r['exact_drop']:.2f}",
                    f"{r['soft@0.00']:.2f}",
                    f"{r['soft@max']:.2f}",
                    f"{r['soft_mean']:.2f}",
                    f"{r['soft_drop']:.2f}",
                ]
            )
            + " |"
        )

    best_soft = sum_df.iloc[0]["method"]
    best_exact = sum_df.sort_values("exact_mean", ascending=False).iloc[0]["method"]
    most_stable_soft = sum_df.sort_values("soft_drop", ascending=True).iloc[0]["method"]
    most_stable_exact = sum_df.sort_values("exact_drop", ascending=True).iloc[0]["method"]

    analysis = (
        "## 论文分析段落（可直接改写使用）\n"
        f"在 complex50 子集与高斯噪声 0.00→{levels[-1]:.2f} 的鲁棒性评测中，"
        "不同推理范式呈现出明显分化：从平均性能看，soft_acc 最优方法为 "
        f"**{best_soft}**，而 exact_acc 最优方法为 **{best_exact}**。"
        "进一步观察噪声前后降幅，"
        f"**{most_stable_soft}** 在软匹配指标上最稳定，"
        f"**{most_stable_exact}** 在精确匹配指标上最稳定，说明其对视觉退化的容忍度更高。"
        "整体上，结构化推理方法在部分噪声区间能够保持较好的语义一致性，"
        "但在严格匹配下仍受细粒度识别误差影响。该结果支持一个结论："
        "在真实场景的噪声扰动下，评价应同时报告 exact_acc 与 soft_acc，"
        "以同时反映答案严格正确性与语义鲁棒性。"
    )

    content = []
    content.append("# Robustness Report (complex50)")
    content.append("")
    content.append("## Exact Accuracy by Noise")
    content.append(md_table(exact))
    content.append("")
    content.append("## Soft Accuracy by Noise")
    content.append(md_table(soft))
    content.append("")
    content.append("## Method Ranking and Robustness Summary")
    content.append("\n".join(rank_lines))
    content.append("")
    content.append(analysis)

    out_path.write_text("\n".join(content), encoding="utf-8")
    print(f"REPORT_WRITTEN: {out_path}")


if __name__ == "__main__":
    main()
