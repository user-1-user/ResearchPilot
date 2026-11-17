import json
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "retrieval_eval_results"

DATASET = ROOT / "retrieval_eval_dataset_v2_en.json"
SOURCE = RESULT_DIR / "retrieval_eval_20261114_182912.json"

EXCLUDED = {
    "Q03", "Q08", "Q09", "Q13", "Q29", "Q30", "Q31",
    "Q32", "Q33", "Q36", "Q37", "Q42", "Q45", "Q49"
}

K = 60
BASE_WEIGHT = 0.75
RERANK_WEIGHT = 0.25


def weighted_rrf(base, rerank):
    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    rerank_rank = {cid: i + 1 for i, cid in enumerate(rerank)}

    candidates = list(dict.fromkeys(base + rerank))
    scores = {}

    for cid in candidates:
        score = 0.0
        if cid in base_rank:
            score += BASE_WEIGHT / (K + base_rank[cid])
        if cid in rerank_rank:
            score += RERANK_WEIGHT / (K + rerank_rank[cid])
        scores[cid] = score

    return sorted(candidates, key=lambda cid: scores[cid], reverse=True)


def selective_gate(base, rerank):
    if not base or not rerank:
        return False

    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    rerank_rank = {cid: i + 1 for i, cid in enumerate(rerank)}

    reranker_top1 = rerank[0]
    baseline_top1 = base[0]

    # 只有满足两个条件才启用融合：
    # 1. Reranker 第一名原本就在 RRF 前三名；
    # 2. RRF 第一名被 Reranker 排到前三名之外。
    return (
        base_rank.get(reranker_top1, 9999) <= 3
        and rerank_rank.get(baseline_top1, 9999) > 3
    )


def evaluate(rankings, gold_ids):
    gold = set(gold_ids)
    if not gold:
        raise ValueError("Empty gold_ids")

    hits5 = len(set(rankings[:5]) & gold)
    hits10 = len(set(rankings[:10]) & gold)

    first_rank = next(
        (i for i, cid in enumerate(rankings, 1) if cid in gold),
        None
    )

    return {
        "recall@5": hits5 / len(gold),
        "recall@10": hits10 / len(gold),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "first_gold_rank": first_rank,
        "top5_gold_count": hits5,
    }


def aggregate(rows, method):
    selected = [r for r in rows if r["method"] == method]
    n = len(selected)

    return {
        "questions": n,
        "recall@5": sum(x["recall@5"] for x in selected) / n,
        "recall@10": sum(x["recall@10"] for x in selected) / n,
        "mrr": sum(x["mrr"] for x in selected) / n,
        "rank1_count": sum(
            1 for x in selected if x["first_gold_rank"] == 1
        ),
        "top5_count": sum(
            1 for x in selected if x["recall@5"] > 0
        ),
    }


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))

    questions = {
        q["id"]: q for q in dataset["questions"]
        if q["id"] not in EXCLUDED
    }
    source_rows = {
        row["id"]: row for row in source["per_question_results"]
        if row["id"] not in EXCLUDED
    }

    if len(questions) != 36:
        raise RuntimeError(f"Expected 36 held-out questions, got {len(questions)}")

    if set(questions) != set(source_rows):
        raise RuntimeError("Dataset/source question IDs do not match")

    results = []
    selected_questions = []

    for qid in sorted(questions):
        question = questions[qid]
        source_row = source_rows[qid]

        gold_ids = question["relevant_chunk_ids"]
        rankings = source_row["rankings"]

        if "RRF" not in rankings:
            raise KeyError(f"{qid}: missing RRF ranking")
        if "RRF_plus_BGE_Reranker" not in rankings:
            raise KeyError(f"{qid}: missing reranker ranking")

        base = rankings["RRF"]
        rerank = rankings["RRF_plus_BGE_Reranker"]

        if not base or not rerank:
            raise RuntimeError(f"{qid}: empty ranking")

        full_hybrid = weighted_rrf(base, rerank)
        use_hybrid = selective_gate(base, rerank)
        selective = full_hybrid if use_hybrid else base

        if use_hybrid:
            selected_questions.append(qid)

        methods = {
            "BASELINE": base,
            "FULL_HYBRID": full_hybrid,
            "SELECTIVE_HYBRID": selective,
        }

        for method, ranked_ids in methods.items():
            metrics = evaluate(ranked_ids, gold_ids)
            results.append({
                "question_id": qid,
                "topic": question["topic"],
                "method": method,
                "selected_for_rerank": use_hybrid,
                "candidate_count": len(set(base) | set(rerank)),
                **metrics,
            })

    methods = ["BASELINE", "FULL_HYBRID", "SELECTIVE_HYBRID"]
    summaries = {m: aggregate(results, m) for m in methods}

    print("\n" + "=" * 88)
    print("36-QUESTION HELD-OUT SELECTIVE RERANKING EXPERIMENT")
    print("=" * 88)
    print("Excluded:", len(EXCLUDED))
    print("Held-out questions:", len(questions))
    print("Selection rule: reranker top1 in baseline top3 AND baseline top1 reranker rank > 3")
    print("Selected questions:", len(selected_questions), "/", len(questions))
    print("Selected IDs:", selected_questions)

    print("\n{:<20} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "METHOD", "R@5", "R@10", "MRR", "Rank1", "Top5"
    ))

    for method in methods:
        s = summaries[method]
        print("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>9} {:>9}".format(
            method, s["recall@5"], s["recall@10"],
            s["mrr"], s["rank1_count"], s["top5_count"]
        ))

    print("\nPER-QUESTION CHANGES: SELECTIVE vs BASELINE")
    baseline_map = {
        r["question_id"]: r for r in results
        if r["method"] == "BASELINE"
    }
    selective_map = {
        r["question_id"]: r for r in results
        if r["method"] == "SELECTIVE_HYBRID"
    }

    for qid in sorted(questions):
        b = baseline_map[qid]
        s = selective_map[qid]
        if (
            b["first_gold_rank"] != s["first_gold_rank"]
            or b["recall@10"] != s["recall@10"]
        ):
            print(
                f"{qid}: rank {b['first_gold_rank']} -> {s['first_gold_rank']}, "
                f"R@10 {b['recall@10']:.3f} -> {s['recall@10']:.3f}, "
                f"selected={s['selected_for_rerank']}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"selective_rerank_36q_{timestamp}.json"
    csv_path = RESULT_DIR / f"selective_rerank_36q_{timestamp}.csv"

    output = {
        "experiment": "36-question held-out selective reranking",
        "created_at": timestamp,
        "excluded_question_ids": sorted(EXCLUDED),
        "held_out_question_count": len(questions),
        "selection_rule": {
            "reranker_top1_in_baseline_top3": True,
            "baseline_top1_reranker_rank_greater_than": 3,
        },
        "fusion": {
            "baseline_weight": BASE_WEIGHT,
            "reranker_weight": RERANK_WEIGHT,
            "rrf_k": K,
        },
        "selected_question_ids": selected_questions,
        "aggregate": summaries,
        "results": results,
    }

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:")
    print(json_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
