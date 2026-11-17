# -*- coding: utf-8 -*-
"""
Paired statistical analysis of the saved 16-question language ablation.
Run from backend:
    .\venv\Scripts\python.exe .\tests\analyze_language_ablation.py

Reads the existing language_ablation_20261104_231314.json.
Writes a new timestamped statistics report; does not modify the experiment result.
No model loading and no database access.
"""
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
import random

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "tests" / "retrieval_eval_results" / "language_ablation_20261104_231314.json"
OUTPUT_DIR = ROOT / "tests" / "retrieval_eval_results"
BOOTSTRAP_REPS = 100_000
RANDOM_SEED = 20261104

METRICS = ("recall_at_5", "recall_at_10", "mrr")
COMPARISONS = (
    ("bilingual", "zh", "中英双语 - 中文"),
    ("bilingual", "en", "中英双语 - 英文"),
)
ARM_LABELS = {"zh": "中文", "en": "英文", "bilingual": "中英双语"}
METRIC_LABELS = {
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "mrr": "MRR",
}


def load_report():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到第一轮实验结果：{INPUT_PATH}")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    for arm in ("zh", "en", "bilingual"):
        if arm not in report.get("arms", {}):
            raise ValueError(f"实验结果缺少组别：{arm}")
    return report


def derive_question_metrics(report):
    """Reconstruct per-question metrics from stored full rankings and Gold ranks."""
    result = {}
    coverage = report["stage1_candidate_coverage"]["per_question"]

    for arm in ("zh", "en", "bilingual"):
        result[arm] = {}
        questions = report["arms"][arm]["questions"]

        for qid, item in questions.items():
            gold_total = coverage[qid]["gold_total"]
            if gold_total <= 0:
                raise ValueError(f"{qid} 的 Gold 总数异常：{gold_total}")

            gold_ranks = item["gold_ranks"]
            first_rank = item["first_gold_rank"]

            result[arm][qid] = {
                "recall_at_5": sum(rank <= 5 for rank in gold_ranks) / gold_total,
                "recall_at_10": sum(rank <= 10 for rank in gold_ranks) / gold_total,
                "mrr": 1.0 / first_rank if first_rank else 0.0,
                "gold_total": gold_total,
                "gold_ranks": gold_ranks,
                "first_gold_rank": first_rank,
            }

    qids = sorted(result["zh"])
    if len(qids) != 16:
        raise ValueError(f"预期 16 道题，实际得到 {len(qids)} 道。")
    if any(set(result[arm]) != set(qids) for arm in result):
        raise ValueError("三组题目 ID 不一致。")
    return result, qids


def bootstrap_ci(differences, reps=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    """Percentile 95% paired bootstrap CI over question-level differences."""
    rng = random.Random(seed)
    n = len(differences)
    samples = []
    for _ in range(reps):
        draw = [differences[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo = samples[int(0.025 * reps)]
    hi = samples[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def exact_paired_permutation_p(differences):
    """Two-sided exact paired sign-flip permutation test over 16 question units."""
    observed = abs(mean(differences))
    extreme = 0
    total = 0
    n = len(differences)
    # Enumerate all 2^n sign assignments; exact for this paired randomization test.
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        permuted_mean = abs(sum(d * s for d, s in zip(differences, signs)) / n)
        if permuted_mean >= observed - 1e-12:
            extreme += 1
        total += 1
    return extreme / total


def rounded(value):
    return round(float(value), 6)


def main():
    report = load_report()
    per_question, qids = derive_question_metrics(report)

    print("=" * 76)
    print("16 题三组 Reranker 实验：逐题与配对统计分析")
    print("=" * 76)
    print(f"输入文件：{INPUT_PATH}")
    print(f"分析单位：{len(qids)} 道题（配对比较）")
    print(f"Bootstrap：{BOOTSTRAP_REPS:,} 次，95% percentile CI")
    print("显著性检验：双侧 exact paired sign-flip permutation test")
    print("\n一、逐题指标")
    print(f"{'题目':<7}{'指标':<12}{'中文':>12}{'英文':>12}{'中英双语':>14}")

    per_question_rows = {}
    for qid in qids:
        per_question_rows[qid] = {}
        for metric in METRICS:
            vals = {arm: per_question[arm][qid][metric] for arm in ARM_LABELS}
            per_question_rows[qid][metric] = vals
            print(
                f"{qid:<7}{METRIC_LABELS[metric]:<12}"
                f"{vals['zh']:>12.4f}{vals['en']:>12.4f}{vals['bilingual']:>14.4f}"
            )

    print("\n二、各组宏平均（由逐题指标重新计算）")
    group_means = {}
    for arm in ARM_LABELS:
        group_means[arm] = {
            metric: mean(per_question[arm][qid][metric] for qid in qids)
            for metric in METRICS
        }
        m = group_means[arm]
        print(
            f"{ARM_LABELS[arm]:<8}"
            f"Recall@5={m['recall_at_5']:.4f}  "
            f"Recall@10={m['recall_at_10']:.4f}  "
            f"MRR={m['mrr']:.4f}"
        )

    comparisons_out = {}
    print("\n三、双语组配对提升与统计不确定性")
    print("差值 = 双语指标 - 对照组指标；CI 不跨 0 表示该区间内差值方向较稳定，")
    print("但小样本结果仍需谨慎解释。p 值仅作探索性参考。")

    for bilingual_arm, comparator_arm, comparison_label in COMPARISONS:
        comp_key = f"{bilingual_arm}_vs_{comparator_arm}"
        comparisons_out[comp_key] = {
            "label": comparison_label,
            "metrics": {}
        }
        print(f"\n{comparison_label}")
        for metric in METRICS:
            diffs = [
                per_question[bilingual_arm][qid][metric]
                - per_question[comparator_arm][qid][metric]
                for qid in qids
            ]
            avg_diff = mean(diffs)
            ci_low, ci_high = bootstrap_ci(
                diffs,
                seed=RANDOM_SEED + len(comparisons_out) * 100 + METRICS.index(metric)
            )
            p_value = exact_paired_permutation_p(diffs)
            wins = sum(d > 1e-12 for d in diffs)
            ties = sum(abs(d) <= 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)

            comparisons_out[comp_key]["metrics"][metric] = {
                "mean_paired_difference": avg_diff,
                "bootstrap_95_ci": [ci_low, ci_high],
                "exact_two_sided_p_value": p_value,
                "question_wins": wins,
                "question_ties": ties,
                "question_losses": losses,
                "per_question_differences": {
                    qid: per_question[bilingual_arm][qid][metric]
                    - per_question[comparator_arm][qid][metric]
                    for qid in qids
                }
            }
            print(
                f"  {METRIC_LABELS[metric]}: 平均差={avg_diff:+.4f}, "
                f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}], "
                f"精确双侧 p={p_value:.5f}, "
                f"逐题胜/平/负={wins}/{ties}/{losses}"
            )

    # Per-question first-relevant-rank comparison, useful for interpreting MRR.
    rank_summary = {}
    for comparator_arm in ("zh", "en"):
        better = same = worse = 0
        for qid in qids:
            b = per_question["bilingual"][qid]["first_gold_rank"]
            c = per_question[comparator_arm][qid]["first_gold_rank"]
            # None means no relevant item retrieved in the ranked list (treated as infinity).
            br = b if b is not None else math.inf
            cr = c if c is not None else math.inf
            if br < cr:
                better += 1
            elif br == cr:
                same += 1
            else:
                worse += 1
        rank_summary[f"bilingual_vs_{comparator_arm}"] = {
            "bilingual_first_gold_rank_better": better,
            "same": same,
            "bilingual_first_gold_rank_worse": worse
        }

    output = {
        "analysis": "Paired per-question statistical analysis of saved 16-question reranker ablation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_file": str(INPUT_PATH),
        "n_questions": len(qids),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPS,
            "confidence_level": 0.95,
            "method": "percentile bootstrap over paired question-level differences",
            "random_seed": RANDOM_SEED
        },
        "permutation_test": {
            "method": "exact two-sided paired sign-flip permutation test",
            "number_of_sign_assignments": 2 ** len(qids),
            "interpretation": "Exploratory; questions may be topically related and n=16 is small."
        },
        "metric_definitions": {
            "recall_at_k": "Per question: number of annotated Gold IDs ranked in top k divided by all annotated Gold IDs for that question.",
            "mrr": "Per question: reciprocal rank of the first Gold result; zero if none is ranked.",
            "paired_difference": "Per-question bilingual metric minus comparator metric, averaged over the same 16 questions."
        },
        "group_macro_means": group_means,
        "per_question_metrics": per_question_rows,
        "comparisons": comparisons_out,
        "first_gold_rank_comparison": rank_summary,
        "limitations": [
            "Only 16 questions were evaluated; confidence intervals and p-values are exploratory.",
            "Several questions concern related Transformer topics, so question-level independence is imperfect.",
            "This analysis evaluates reranking on fixed Stage1 candidate sets; it does not measure retrieval-stage improvements.",
            "The Q02 Gold chunk absent from Stage1 candidates remains in the original Recall denominator."
        ],
        "safety": {
            "experiment_result_read_only": True,
            "database_accessed": False,
            "model_loaded": False,
            "original_result_modified": False
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"language_ablation_statistics_{stamp}.json"
    if output_path.exists():
        raise FileExistsError(f"为避免覆盖，停止写入：{output_path}")
    with output_path.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n四、首个 Gold 排名比较（双语相对对照组）")
    for key, item in rank_summary.items():
        comparator = "中文" if key.endswith("_zh") else "英文"
        print(
            f"  双语 vs {comparator}：双语排名更靠前 "
            f"{item['bilingual_first_gold_rank_better']} 题，"
            f"相同 {item['same']} 题，更靠后 "
            f"{item['bilingual_first_gold_rank_worse']} 题"
        )

    print("\n分析完成。")
    print(f"统计报告已保存：{output_path}")
    print("原始实验 JSON 未修改；未访问数据库；未加载模型。")


if __name__ == "__main__":
    main()
