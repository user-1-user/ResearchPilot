import json
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests/retrieval_eval_results"

SOURCE = DATA_DIR / "query_rewrite_14_20261114_191253.json"
FUSION = DATA_DIR / "query_fusion_14q_20261114_193925.json"

K = 60


def rank_metrics(ranked, gold):
    gold = set(gold)
    first = next(
        (i for i, cid in enumerate(ranked, 1) if cid in gold),
        None
    )
    return {
        "first_gold_rank": first,
        "recall@5": len(set(ranked[:5]) & gold) / len(gold),
        "recall@10": len(set(ranked[:10]) & gold) / len(gold),
        "mrr": 1 / first if first else 0.0,
    }


def weighted_rrf(base_rank, rerank, base_weight, rerank_weight):
    scores = {cid: 0.0 for cid in base_rank}

    for weight, ranking in [
        (base_weight, base_rank),
        (rerank_weight, rerank),
    ]:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] += weight / (K + rank)

    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    fusion = json.loads(FUSION.read_text(encoding="utf-8"))

    fusion_by_q = {}
    for row in fusion["results"]:
        fusion_by_q.setdefault(row["question_id"], {})[row["method"]] = row

    methods = [
        "BASELINE",
        "MEAN",
        "RERANK_RRF",
        "HYBRID_MEAN_25",
        "HYBRID_MEAN_50",
        "HYBRID_MEAN_75",
        "HYBRID_MEAN_90",
        "HYBRID_RRF_50",
        "HYBRID_RRF_75",
    ]

    output = []

    for row in source["results"]:
        qid = row["id"]
        gold = row["gold_ids"]
        base = row["original_rankings"]["RRF"]
        records = fusion_by_q[qid]

        mean_rank = records["MEAN"]["ranked_ids"]
        rerank_rrf = records["RRF"]["ranked_ids"]

        rankings = {
            "BASELINE": base,
            "MEAN": mean_rank,
            "RERANK_RRF": rerank_rrf,
            "HYBRID_MEAN_25": weighted_rrf(base, mean_rank, .25, .75),
            "HYBRID_MEAN_50": weighted_rrf(base, mean_rank, .50, .50),
            "HYBRID_MEAN_75": weighted_rrf(base, mean_rank, .75, .25),
            "HYBRID_MEAN_90": weighted_rrf(base, mean_rank, .90, .10),
            "HYBRID_RRF_50": weighted_rrf(base, rerank_rrf, .50, .50),
            "HYBRID_RRF_75": weighted_rrf(base, rerank_rrf, .75, .25),
        }

        for method, ranked in rankings.items():
            m = rank_metrics(ranked, gold)
            output.append({
                "question_id": qid,
                "method": method,
                **m,
                "top5": ranked[:5],
            })

        print("\n", qid)
        for method in methods:
            item = next(
                x for x in output
                if x["question_id"] == qid and x["method"] == method
            )
            print(
                f"{method:18s} "
                f"Rank={str(item['first_gold_rank']):>3s} "
                f"R@5={item['recall@5']:.3f} "
                f"MRR={item['mrr']:.3f}"
            )

    aggregate = {}

    for method in methods:
        records = [x for x in output if x["method"] == method]
        n = len(records)
        aggregate[method] = {
            "count": n,
            "recall@5": sum(x["recall@5"] for x in records) / n,
            "recall@10": sum(x["recall@10"] for x in records) / n,
            "mrr": sum(x["mrr"] for x in records) / n,
            "rank1": sum(x["first_gold_rank"] == 1 for x in records),
            "top5": sum(
                x["first_gold_rank"] is not None
                and x["first_gold_rank"] <= 5 for x in records
            ),
        }

    print("\n" + "=" * 85)
    print("AGGREGATE")
    print(
        f"{'METHOD':18s} {'R@5':>8s} {'R@10':>8s} "
        f"{'MRR':>8s} {'Rank1':>7s} {'Top5':>7s}"
    )

    for method, m in aggregate.items():
        print(
            f"{method:18s} {m['recall@5']:8.4f} "
            f"{m['recall@10']:8.4f} {m['mrr']:8.4f} "
            f"{m['rank1']:7d} {m['top5']:7d}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = DATA_DIR / f"hybrid_fusion_14q_{timestamp}.json"
    csv_path = DATA_DIR / f"hybrid_fusion_14q_{timestamp}.csv"

    result = {
        "experiment": "14-question weighted hybrid ranking",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "fusion_source": FUSION.name,
        "rrf_k": K,
        "aggregate": aggregate,
        "results": output,
    }

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id", "method", "first_gold_rank",
                "recall@5", "recall@10", "mrr", "top5"
            ],
        )
        writer.writeheader()
        for item in output:
            item = dict(item)
            item["top5"] = json.dumps(item["top5"])
            writer.writerow(item)

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
