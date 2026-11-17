import json
import csv
from pathlib import Path
from datetime import datetime
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "tests" / "cross_paper_eval" / "v6" / "results_corrected"

STAGE1 = RESULT_DIR / "cross_paper_stage1_retrieval.json"
STAGE2 = RESULT_DIR / "cross_paper_stage2_rerank.json"

K = 60
BASE_WEIGHT = 0.75
RERANK_WEIGHT = 0.25


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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

    return (
        base_rank.get(rerank[0], 9999) <= 3
        and rerank_rank.get(base[0], 9999) > 3
    )


def evaluate(ranking, gold_ids):
    gold = set(gold_ids)
    if not gold:
        raise RuntimeError("Empty gold IDs")

    first_rank = next(
        (i for i, cid in enumerate(ranking, 1) if cid in gold),
        None
    )

    return {
        "recall@5": len(set(ranking[:5]) & gold) / len(gold),
        "recall@10": len(set(ranking[:10]) & gold) / len(gold),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "first_gold_rank": first_rank,
        "rank1_hit": bool(ranking and ranking[0] in gold),
    }


def summarize(rows, method):
    selected = [r for r in rows if r["method"] == method]
    return {
        "questions": len(selected),
        "recall@5": mean(r["recall@5"] for r in selected),
        "recall@10": mean(r["recall@10"] for r in selected),
        "mrr": mean(r["mrr"] for r in selected),
        "rank1_count": sum(r["rank1_hit"] for r in selected),
        "top5_count": sum(r["recall@5"] > 0 for r in selected),
    }


def main():
    s1 = load(STAGE1)
    s2 = load(STAGE2)

    if s1.get("question_ids") != s2.get("question_ids"):
        raise RuntimeError("Stage 1/2 question IDs differ")

    ids = s1["question_ids"]
    if len(ids) != 80 or len(set(ids)) != 80:
        raise RuntimeError(f"Expected 80 unique questions, got {len(ids)}")

    rows = []
    selected_ids = []
    selected_by_paper = {}

    for qid in ids:
        a = s1["results"][qid]
        b = s2["results"][qid]

        base = a["rrf_candidate_ids"]
        rerank = b["reranker_ranking"]
        gold = a["gold_chunk_ids"]

        if set(base) != set(rerank) or len(base) != len(rerank):
            raise RuntimeError(f"{qid}: candidate pool mismatch")

        full = weighted_rrf(base, rerank)
        use_hybrid = selective_gate(base, rerank)
        selective = full if use_hybrid else base

        if use_hybrid:
            selected_ids.append(qid)
            pid = a["paper_id"]
            selected_by_paper[pid] = selected_by_paper.get(pid, 0) + 1

        methods = {
            "BASELINE": base,
            "FULL_HYBRID": full,
            "SELECTIVE_HYBRID": selective,
        }

        for method, ranking in methods.items():
            rows.append({
                "question_id": qid,
                "paper_id": a["paper_id"],
                "paper_title": a["paper_title"],
                "candidate_count": len(base),
                "selected_for_rerank": use_hybrid,
                "method": method,
                **evaluate(ranking, gold),
            })

    methods = ["BASELINE", "FULL_HYBRID", "SELECTIVE_HYBRID"]
    summaries = {m: summarize(rows, m) for m in methods}

    print("\\n" + "=" * 88)
    print("V6 SELECTIVE RERANKING — FIXED 36Q GATE ON 80 QUESTIONS")
    print("=" * 88)
    print("Gate: reranker top1 in RRF top3 AND RRF top1 reranker rank > 3")
    print(f"Selected: {len(selected_ids)}/80 ({len(selected_ids)/80:.2%})")
    print("Selected by paper:", selected_by_paper)

    print("\\n{:<20} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "METHOD", "R@5", "R@10", "MRR", "Rank1", "Top5"
    ))

    for method in methods:
        x = summaries[method]
        print("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>9} {:>9}".format(
            method, x["recall@5"], x["recall@10"], x["mrr"],
            x["rank1_count"], x["top5_count"]
        ))

    print("\\nSELECTIVE vs BASELINE")
    for key in ("recall@5", "recall@10", "mrr"):
        delta = (
            summaries["SELECTIVE_HYBRID"][key]
            - summaries["BASELINE"][key]
        )
        print(f"{key}: {delta:+.6f}")

    print("\\nSELECTIVE vs FULL HYBRID")
    for key in ("recall@5", "recall@10", "mrr"):
        delta = (
            summaries["SELECTIVE_HYBRID"][key]
            - summaries["FULL_HYBRID"][key]
        )
        print(f"{key}: {delta:+.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "V6 selective reranking with fixed 36q gate",
        "created_at": timestamp,
        "question_count": 80,
        "selection_rule": {
            "reranker_top1_in_baseline_top3": True,
            "baseline_top1_reranker_rank_greater_than": 3,
        },
        "selected_question_ids": selected_ids,
        "selected_by_paper": selected_by_paper,
        "aggregate": summaries,
        "results": rows,
    }

    json_path = RESULT_DIR / f"selective_rerank_v6_80q_{timestamp}.json"
    csv_path = RESULT_DIR / f"selective_rerank_v6_80q_{timestamp}.csv"

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
