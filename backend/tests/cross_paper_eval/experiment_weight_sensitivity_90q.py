import json
import csv
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = BASE_DIR / "tests" / "cross_paper_eval" / "results"

DATASET_PATH = (
    BASE_DIR / "tests" / "cross_paper_eval"
    / "cross_paper_eval_dataset_v4_reviewed.json"
)
STAGE1_PATH = RESULT_DIR / "cross_paper_stage1_retrieval.json"
STAGE2_PATH = RESULT_DIR / "cross_paper_stage2_rerank.json"

K = 60
BOOTSTRAP_N = 10000
SEED = 20261115

# CE 权重；RRF 权重自动等于 1 - CE 权重
WEIGHTS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

PAPERS = {
    "4cd8e5914040": "GPT-3",
    "20c0b7cbf888": "ViT",
    "aa7deee51d3b": "Chemotactic",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rank_fusion(base, rerank, ce_weight):
    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    ce_rank = {cid: i + 1 for i, cid in enumerate(rerank)}

    candidates = list(dict.fromkeys(base + rerank))
    base_weight = 1.0 - ce_weight

    scores = {}
    for cid in candidates:
        score = 0.0
        if cid in base_rank:
            score += base_weight / (K + base_rank[cid])
        if cid in ce_rank:
            score += ce_weight / (K + ce_rank[cid])
        scores[cid] = score

    # 保留稳定排序，确保并列时行为一致
    return sorted(candidates, key=lambda cid: scores[cid], reverse=True)


def question_metrics(ranking, gold_ids):
    gold = set(gold_ids)
    positions = [
        i + 1 for i, cid in enumerate(ranking) if cid in gold
    ]
    first = min(positions) if positions else None

    return {
        "recall@5": int(first is not None and first <= 5),
        "recall@10": int(first is not None and first <= 10),
        "mrr": 1.0 / first if first is not None else 0.0,
        "first_gold_rank": first if first is not None else 0,
    }


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot aggregate empty rows")

    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in ("recall@5", "recall@10", "mrr")
    }


def macro_metrics(rows):
    by_paper = defaultdict(list)
    for row in rows:
        by_paper[row["paper_id"]].append(row)

    per_paper = {
        paper_id: aggregate(items)
        for paper_id, items in by_paper.items()
    }

    macro = {
        metric: sum(v[metric] for v in per_paper.values())
        / len(per_paper)
        for metric in ("recall@5", "recall@10", "mrr")
    }
    return macro, per_paper


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    high = min(low + 1, len(values) - 1)
    fraction = index - low
    return values[low] * (1 - fraction) + values[high] * fraction


def bootstrap_delta(rows, baseline_name, method_name):
    by_paper = defaultdict(list)
    for row in rows:
        by_paper[row["paper_id"]].append(row)

    paper_ids = sorted(by_paper)
    if len(paper_ids) != 3:
        raise RuntimeError(f"Expected 3 papers, got {len(paper_ids)}")

    for pid in paper_ids:
        if len(by_paper[pid]) != 30:
            raise RuntimeError(
                f"Expected 30 questions for {PAPERS[pid]}, "
                f"got {len(by_paper[pid])}"
            )

    rng = random.Random(SEED)
    output = {}

    for metric in ("recall@5", "recall@10", "mrr"):
        differences = []

        for _ in range(BOOTSTRAP_N):
            paper_diffs = []

            for pid in paper_ids:
                sample = [
                    rng.choice(by_paper[pid])
                    for _ in range(len(by_paper[pid]))
                ]

                baseline = sum(
                    x[baseline_name][metric] for x in sample
                ) / len(sample)

                method = sum(
                    x[method_name][metric] for x in sample
                ) / len(sample)

                paper_diffs.append(method - baseline)

            differences.append(sum(paper_diffs) / len(paper_diffs))

        output[metric] = {
            "mean_difference": sum(differences) / len(differences),
            "ci_95_low": percentile(differences, 0.025),
            "ci_95_high": percentile(differences, 0.975),
            "probability_difference_gt_zero": (
                sum(x > 0 for x in differences) / len(differences)
            ),
        }

    return output


def main():
    dataset = load_json(DATASET_PATH)
    stage1 = load_json(STAGE1_PATH)
    stage2 = load_json(STAGE2_PATH)

    if stage1.get("dataset_fingerprint") != stage2.get("dataset_fingerprint"):
        raise RuntimeError("Stage1/Stage2 dataset fingerprints do not match")

    questions = dataset["questions"]
    retrieval_results = stage1["results"]
    rerank_results = stage2["results"]

    if len(questions) != 90:
        raise RuntimeError(f"Expected 90 dataset questions, got {len(questions)}")

    if len(retrieval_results) != 90 or len(rerank_results) != 90:
        raise RuntimeError("Stage cache does not contain exactly 90 questions")

    methods = {
        "BASELINE_RRF": 0.0,
        "RRF_90_CE_10": 0.1,
        "RRF_80_CE_20": 0.2,
        "RRF_75_CE_25": 0.25,
        "RRF_70_CE_30": 0.3,
        "RRF_60_CE_40": 0.4,
        "RRF_50_CE_50": 0.5,
        "PURE_CE": 1.0,
    }

    detail_rows = []

    for q in questions:
        qid = q["question_id"]
        pid = q["paper_id"]

        if pid not in PAPERS:
            raise RuntimeError(f"Unexpected paper ID: {pid}")

        if qid not in retrieval_results or qid not in rerank_results:
            raise RuntimeError(f"Missing cached result: {qid}")

        base = retrieval_results[qid]["rrf_candidate_ids"]
        rerank = rerank_results[qid]["reranker_ranking"]
        gold = q["gold_chunk_ids"]

        if set(base) != set(rerank):
            raise RuntimeError(f"Candidate mismatch for {qid}")

        if not gold:
            raise RuntimeError(f"No Gold chunk for {qid}")

        row = {
            "question_id": qid,
            "paper_id": pid,
            "paper": PAPERS[pid],
            "question": q["question"],
            "gold_chunk_ids": "|".join(gold),
        }

        for name, ce_weight in methods.items():
            ranking = rank_fusion(base, rerank, ce_weight)
            row[name] = question_metrics(ranking, gold)

        detail_rows.append(row)

    # 统计各方法整体及分论文指标
    summary = {}
    print("\n" + "=" * 100)
    print("WEIGHT SENSITIVITY: 90 QUESTIONS / 3 PAPERS")
    print("=" * 100)
    print(
        f"{'METHOD':<18} {'R@5':>9} {'R@10':>9} "
        f"{'MRR':>9} {'IMPROVED':>10} {'WORSE':>8} {'UNCHANGED':>11}"
    )

    baseline_rows = [r["BASELINE_RRF"] for r in detail_rows]

    for name in methods:
        current_rows = [r[name] for r in detail_rows]
        macro, per_paper = macro_metrics([
            {
                "paper_id": r["paper_id"],
                **r[name],
            }
            for r in detail_rows
        ])

        improved = sum(
            r[name]["mrr"] > r["BASELINE_RRF"]["mrr"]
            for r in detail_rows
        )
        worsened = sum(
            r[name]["mrr"] < r["BASELINE_RRF"]["mrr"]
            for r in detail_rows
        )
        unchanged = 90 - improved - worsened

        summary[name] = {
            "ce_weight": methods[name],
            "rrf_weight": 1 - methods[name],
            "macro": macro,
            "per_paper": {
                PAPERS[pid]: metrics
                for pid, metrics in per_paper.items()
            },
            "question_changes_vs_baseline": {
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            },
        }

        print(
            f"{name:<18} "
            f"{macro['recall@5']:>9.4f} "
            f"{macro['recall@10']:>9.4f} "
            f"{macro['mrr']:>9.4f} "
            f"{improved:>10} {worsened:>8} {unchanged:>11}"
        )

    # Bootstrap：所有方案与纯 RRF baseline 配对比较
    bootstrap = {}
    print("\n" + "=" * 100)
    print("STRATIFIED PAIRED BOOTSTRAP: METHOD - BASELINE_RRF")
    print("=" * 100)

    for name in methods:
        if name == "BASELINE_RRF":
            continue

        result = bootstrap_delta(detail_rows, "BASELINE_RRF", name)
        bootstrap[name] = result

        print(f"\n{name}")
        for metric, values in result.items():
            print(
                f"  {metric}: diff={values['mean_difference']:+.5f}, "
                f"95% CI=[{values['ci_95_low']:+.5f}, "
                f"{values['ci_95_high']:+.5f}], "
                f"P(diff>0)={values['probability_difference_gt_zero']:.4f}"
            )

    # 各论文单独指标
    print("\n" + "=" * 100)
    print("PER-PAPER METRICS")
    print("=" * 100)

    for name in methods:
        print(f"\n{name}")
        for paper_name, metrics in summary[name]["per_paper"].items():
            print(
                f"  {paper_name:<14} "
                f"R@5={metrics['recall@5']:.4f} "
                f"R@10={metrics['recall@10']:.4f} "
                f"MRR={metrics['mrr']:.4f}"
            )

    # 重点题目排名变化
    focus_ids = [
        "GPT3_28",
        "CHEMOTACTIC_28",
        "CHEMOTACTIC_21",
        "CHEMOTACTIC_09",
        "VIT_08",
        "VIT_14",
        "VIT_16",
        "VIT_17",
    ]

    print("\n" + "=" * 100)
    print("FOCUS QUESTION RANKS")
    print("=" * 100)

    for row in detail_rows:
        if row["question_id"] not in focus_ids:
            continue

        print("\n", row["question_id"], "-", row["question"])
        for name in methods:
            print(
                f"  {name:<18} "
                f"rank={row[name]['first_gold_rank']}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"weight_sensitivity_90q_{timestamp}.json"
    csv_path = RESULT_DIR / f"weight_sensitivity_90q_{timestamp}.csv"

    output = {
        "experiment": "cached_weight_sensitivity_90_questions",
        "created_at": timestamp,
        "dataset_fingerprint": stage1["dataset_fingerprint"],
        "question_count": len(detail_rows),
        "paper_count": len(PAPERS),
        "weights": methods,
        "summary": summary,
        "bootstrap_vs_baseline": bootstrap,
        "limitations": [
            "Weights are exploratory and evaluated on the same 90 questions.",
            "Only three papers, one per research field, were used.",
            "Bootstrap resamples questions within papers, not independent papers.",
            "Gold evaluation uses the supplied gold_chunk_ids.",
            "No model inference or production code modification was performed.",
        ],
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    columns = [
        "question_id", "paper_id", "paper",
        "question", "gold_chunk_ids",
    ]
    for name in methods:
        columns.extend([
            f"{name}_rank",
            f"{name}_recall5",
            f"{name}_recall10",
            f"{name}_mrr",
        ])

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in detail_rows:
            record = {
                "question_id": row["question_id"],
                "paper_id": row["paper_id"],
                "paper": row["paper"],
                "question": row["question"],
                "gold_chunk_ids": row["gold_chunk_ids"],
            }

            for name in methods:
                record[f"{name}_rank"] = row[name]["first_gold_rank"]
                record[f"{name}_recall5"] = row[name]["recall@5"]
                record[f"{name}_recall10"] = row[name]["recall@10"]
                record[f"{name}_mrr"] = row[name]["mrr"]

            writer.writerow(record)

    print("\nSAVED:")
    print(json_path)
    print(csv_path)
    print("\nDONE. Original caches and main.py were not modified.")


if __name__ == "__main__":
    main()
