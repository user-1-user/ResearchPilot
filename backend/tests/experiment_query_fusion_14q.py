import sys
import json
import csv
import math
import importlib.util
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import main

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SOURCE = ROOT / "tests/retrieval_eval_results/query_rewrite_14_20261114_191253.json"
OUT_DIR = ROOT / "tests/retrieval_eval_results"
RRF_K = 60


def load_eval_module():
    path = ROOT / "tests/evaluate_retrieval_lowmem_v2.py"
    spec = importlib.util.spec_from_file_location("eval_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank_scores(scores):
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def metrics(ranked, gold_ids, scores):
    gold = set(gold_ids)
    first_rank = next(
        (i for i, cid in enumerate(ranked, 1) if cid in gold),
        None
    )

    recall5 = len(set(ranked[:5]) & gold) / len(gold)
    recall10 = len(set(ranked[:10]) & gold) / len(gold)

    gold_scores = [scores[x] for x in gold if x in scores]
    neg_scores = [scores[x] for x in ranked if x not in gold]

    best_gold = max(gold_scores) if gold_scores else None
    best_neg = max(neg_scores) if neg_scores else None

    return {
        "first_gold_rank": first_rank,
        "recall@5": recall5,
        "recall@10": recall10,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "best_gold_score": best_gold,
        "best_non_gold_score": best_neg,
        "margin": (
            best_gold - best_neg
            if best_gold is not None and best_neg is not None
            else None
        ),
        "top5": [
            {
                "rank": i,
                "chunk_id": cid,
                "score": scores[cid],
                "is_gold": cid in gold,
            }
            for i, cid in enumerate(ranked[:5], 1)
        ],
    }


def main_run():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["results"]

    print("Loading paper chunks...")
    ev = load_eval_module()
    paper, chunks, all_ids = ev.load_paper()
    text_by_id = {x["id"]: x["text"] for x in chunks}

    print("Paper:", paper["filename"])
    print("Chunks:", len(chunks))
    print("Loading CrossEncoder...")
    reranker = main._get_reranker()

    if reranker is None:
        raise RuntimeError("CrossEncoder failed to load")

    output = []

    for row in rows:
        qid = row["id"]
        original_query = row["original_question"]
        rewritten_query = row["rewritten_question"]
        gold_ids = row["gold_ids"]

        # 固定原始检索阶段的 RRF 候选池
        candidates = row["original_rankings"]["RRF"]

        if len(candidates) != 28 or len(set(candidates)) != 28:
            raise RuntimeError(f"{qid}: invalid candidate pool")

        if not set(gold_ids).issubset(set(candidates)):
            raise RuntimeError(f"{qid}: Gold missing from candidate pool")

        if any(cid not in text_by_id for cid in candidates):
            raise RuntimeError(f"{qid}: missing chunk text")

        print("\n" + "=" * 72)
        print(f"{qid} | candidates={len(candidates)}")

        score_maps = {}

        for label, query in [
            ("ORIGINAL", original_query),
            ("REWRITE", rewritten_query),
        ]:
            pairs = [
                [query, text_by_id[cid][:1800]]
                for cid in candidates
            ]

            values = reranker.predict(
                pairs,
                batch_size=1,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            )

            values = [float(x) for x in values]

            if len(values) != len(candidates):
                raise RuntimeError(f"{qid}/{label}: score count mismatch")

            if not all(math.isfinite(x) for x in values):
                raise RuntimeError(f"{qid}/{label}: invalid scores")

            score_maps[label] = dict(zip(candidates, values))

        original_scores = score_maps["ORIGINAL"]
        rewrite_scores = score_maps["REWRITE"]

        # 两条 Query 分数取平均
        mean_scores = {
            cid: (original_scores[cid] + rewrite_scores[cid]) / 2
            for cid in candidates
        }

        # 两条 Query 排名进行 RRF 融合
        rrf_scores = {cid: 0.0 for cid in candidates}

        for label in ["ORIGINAL", "REWRITE"]:
            ranked = rank_scores(score_maps[label])
            for rank, cid in enumerate(ranked, 1):
                rrf_scores[cid] += 1.0 / (RRF_K + rank)

        # 原始检索阶段 RRF 排名，作为参照
        baseline_ranking = row["original_rankings"]["RRF"]
        baseline_scores = {
            cid: float(len(baseline_ranking) - i)
            for i, cid in enumerate(baseline_ranking)
        }

        methods = {
            "ORIGINAL": original_scores,
            "MEAN": mean_scores,
            "RRF": rrf_scores,
            "RRF_BASELINE": baseline_scores,
        }

        for method, scores in methods.items():
            ranked = rank_scores(scores)
            result = metrics(ranked, gold_ids, scores)

            record = {
                "question_id": qid,
                "method": method,
                "candidate_count": len(candidates),
                "gold_ids": gold_ids,
                **result,
                "ranked_ids": ranked,
            }
            output.append(record)

            print(
                f"{method:13s} "
                f"GoldRank={str(result['first_gold_rank']):>4s} "
                f"R@5={result['recall@5']:.4f} "
                f"R@10={result['recall@10']:.4f} "
                f"MRR={result['mrr']:.4f}"
            )

    # 汇总宏平均指标
    aggregate = {}

    for method in ["ORIGINAL", "MEAN", "RRF", "RRF_BASELINE"]:
        records = [x for x in output if x["method"] == method]
        n = len(records)

        aggregate[method] = {
            "question_count": n,
            "recall@5": sum(x["recall@5"] for x in records) / n,
            "recall@10": sum(x["recall@10"] for x in records) / n,
            "mrr": sum(x["mrr"] for x in records) / n,
            "rank1_count": sum(
                x["first_gold_rank"] == 1 for x in records
            ),
            "top5_count": sum(
                x["first_gold_rank"] is not None
                and x["first_gold_rank"] <= 5
                for x in records
            ),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"query_fusion_14q_{timestamp}.json"
    csv_path = OUT_DIR / f"query_fusion_14q_{timestamp}.csv"

    result = {
        "experiment": "14-question two-query fusion",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "candidate_policy": "Fixed original RRF 28 candidates",
        "queries": ["ORIGINAL", "REWRITE"],
        "rrf_k": RRF_K,
        "reranker_settings": {
            "text_limit": 1800,
            "batch_size": 1,
            "activation": "Identity",
        },
        "aggregate": aggregate,
        "results": output,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    columns = [
        "question_id", "method", "candidate_count",
        "first_gold_rank", "recall@5", "recall@10", "mrr",
        "best_gold_score", "best_non_gold_score", "margin",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in output:
            writer.writerow({k: item.get(k) for k in columns})

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print(
        f"{'METHOD':13s} {'R@5':>9s} {'R@10':>9s} "
        f"{'MRR':>9s} {'Rank1':>8s} {'Top5':>8s}"
    )

    for method, item in aggregate.items():
        print(
            f"{method:13s} "
            f"{item['recall@5']:9.4f} "
            f"{item['recall@10']:9.4f} "
            f"{item['mrr']:9.4f} "
            f"{item['rank1_count']:8d} "
            f"{item['top5_count']:8d}"
        )

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main_run()
