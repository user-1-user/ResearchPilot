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

SOURCE = ROOT / "tests/retrieval_eval_results/query_rewrite_variants_5q_20261114_192049.json"
OUT_DIR = ROOT / "tests/retrieval_eval_results"

RRF_K = 60
TOP_KS = (5, 10)

def load_eval_module():
    path = ROOT / "tests/evaluate_retrieval_lowmem_v2.py"
    spec = importlib.util.spec_from_file_location("eval_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def calculate_metrics(ranked_ids, gold_ids, score_by_id):
    gold = set(gold_ids)
    if not gold:
        raise RuntimeError("Empty gold_ids")

    first_rank = None
    for i, cid in enumerate(ranked_ids, 1):
        if cid in gold:
            first_rank = i
            break

    result = {
        "first_gold_rank": first_rank,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "gold_count": len(gold),
    }

    for k in TOP_KS:
        hits = len(set(ranked_ids[:k]) & gold)
        result[f"recall@{k}"] = hits / len(gold)

    gold_scores = [score_by_id[cid] for cid in gold if cid in score_by_id]
    neg_scores = [score_by_id[cid] for cid in ranked_ids if cid not in gold]

    best_gold = max(gold_scores) if gold_scores else None
    best_neg = max(neg_scores) if neg_scores else None

    result["best_gold_score"] = best_gold
    result["best_non_gold_score"] = best_neg
    result["margin"] = (
        best_gold - best_neg
        if best_gold is not None and best_neg is not None
        else None
    )

    result["top5"] = [
        {
            "rank": i,
            "chunk_id": cid,
            "score": score_by_id[cid],
            "is_gold": cid in gold,
        }
        for i, cid in enumerate(ranked_ids[:5], 1)
    ]
    return result

def rank_scores(score_by_id):
    return sorted(
        score_by_id,
        key=lambda cid: score_by_id[cid],
        reverse=True,
    )

def main_run():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source: {SOURCE}")

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    grouped = {}

    # Gold labels are stored in the referenced 14-question source file.
    gold_source_path = SOURCE.parent / source["source_result"]
    if not gold_source_path.exists():
        raise FileNotFoundError(f"Missing Gold source: {gold_source_path}")

    gold_source = json.loads(gold_source_path.read_text(encoding="utf-8"))
    gold_by_qid = {
        row["id"]: row["gold_ids"]
        for row in gold_source["results"]
    }

    for row in source["results"]:
        grouped.setdefault(row["question_id"], {})[row["label"]] = row

    required_labels = [
        "ORIGINAL", "VARIANT_1", "VARIANT_2", "VARIANT_3"
    ]

    for qid, rows in grouped.items():
        missing = [x for x in required_labels if x not in rows]
        if missing:
            raise RuntimeError(f"{qid}: missing labels {missing}")

    print("Loading paper chunks...")
    ev = load_eval_module()
    paper, chunks, all_ids = ev.load_paper()
    text_by_id = {r["id"]: r["text"] for r in chunks}

    print("Paper:", paper["filename"])
    print("Chunks:", len(chunks))
    print("Loading CrossEncoder...")
    reranker = main._get_reranker()

    if reranker is None:
        raise RuntimeError("CrossEncoder failed to load")

    output = []
    aggregate = {}

    for qid in sorted(grouped):
        rows = grouped[qid]
        base = rows["ORIGINAL"]
        candidates = base["candidate_ids"]
        if qid not in gold_by_qid:
            raise RuntimeError(f"{qid}: Gold labels not found in source result")
        gold_ids = gold_by_qid[qid]

        if len(candidates) != 28 or len(set(candidates)) != 28:
            raise RuntimeError(f"{qid}: candidate pool is not 28 unique IDs")

        for cid in candidates:
            if cid not in text_by_id:
                raise RuntimeError(f"{qid}: missing chunk text {cid}")

        score_maps = {}
        query_records = []

        print("\n" + "=" * 72)
        print(qid, "| candidates:", len(candidates))

        for label in required_labels:
            row = rows[label]
            query = row["query"]

            pairs = [
                [query, text_by_id[cid][:1800]]
                for cid in candidates
            ]

            scores = reranker.predict(
                pairs,
                batch_size=1,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            )

            score_list = [float(x) for x in scores]

            if len(score_list) != len(candidates):
                raise RuntimeError(f"{qid}/{label}: score count mismatch")

            if not all(math.isfinite(x) for x in score_list):
                raise RuntimeError(f"{qid}/{label}: invalid score")

            score_map = dict(zip(candidates, score_list))
            score_maps[label] = score_map

            ranked = rank_scores(score_map)
            details = calculate_metrics(ranked, gold_ids, score_map)

            query_records.append({
                "label": label,
                "query": query,
                "metrics": details,
            })

            print(
                f"{label:10s} "
                f"GoldRank={str(details['first_gold_rank']):>4s} "
                f"R@5={details['recall@5']:.4f} "
                f"MRR={details['mrr']:.4f}"
            )

        # MAX: each chunk takes its highest score across four queries.
        max_scores = {
            cid: max(score_maps[label][cid] for label in required_labels)
            for cid in candidates
        }

        # MEAN: each chunk takes the mean score across four queries.
        mean_scores = {
            cid: sum(score_maps[label][cid] for label in required_labels) / len(required_labels)
            for cid in candidates
        }

        # RRF: fuse the four rankings; k=60.
        rrf_scores = {cid: 0.0 for cid in candidates}
        for label in required_labels:
            ranked = rank_scores(score_maps[label])
            for rank, cid in enumerate(ranked, 1):
                rrf_scores[cid] += 1.0 / (RRF_K + rank)

        fusion_maps = {
            "MAX": max_scores,
            "MEAN": mean_scores,
            "RRF": rrf_scores,
        }

        methods = {}

        # Original single-query baseline.
        original_scores = score_maps["ORIGINAL"]
        methods["ORIGINAL"] = original_scores

        for method, score_map in fusion_maps.items():
            methods[method] = score_map

        for method, score_map in methods.items():
            ranked = rank_scores(score_map)
            details = calculate_metrics(ranked, gold_ids, score_map)

            record = {
                "question_id": qid,
                "method": method,
                "candidate_count": len(candidates),
                "gold_ids": gold_ids,
                "first_gold_rank": details["first_gold_rank"],
                "recall@5": details["recall@5"],
                "recall@10": details["recall@10"],
                "mrr": details["mrr"],
                "best_gold_score": details["best_gold_score"],
                "best_non_gold_score": details["best_non_gold_score"],
                "margin": details["margin"],
                "top5": details["top5"],
                "ranked_ids": ranked,
                "scores": score_map,
            }
            output.append(record)

            print(
                f"  {method:8s} "
                f"GoldRank={str(details['first_gold_rank']):>4s} "
                f"R@5={details['recall@5']:.4f} "
                f"R@10={details['recall@10']:.4f} "
                f"MRR={details['mrr']:.4f} "
                f"Margin={details['margin']:.4f}"
            )

    # Macro-average metrics across questions.
    for method in ["ORIGINAL", "MAX", "MEAN", "RRF"]:
        records = [r for r in output if r["method"] == method]
        aggregate[method] = {
            "question_count": len(records),
            "recall@5": sum(r["recall@5"] for r in records) / len(records),
            "recall@10": sum(r["recall@10"] for r in records) / len(records),
            "mrr": sum(r["mrr"] for r in records) / len(records),
            "gold_rank_1_count": sum(
                1 for r in records if r["first_gold_rank"] == 1
            ),
            "gold_rank_5_count": sum(
                1 for r in records
                if r["first_gold_rank"] is not None
                and r["first_gold_rank"] <= 5
            ),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"query_fusion_5q_{timestamp}.json"
    csv_path = OUT_DIR / f"query_fusion_5q_{timestamp}.csv"

    result = {
        "experiment": "Five-question multi-query fusion",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "candidate_policy": "Original 28 candidates fixed per question",
        "query_labels": required_labels,
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
        "top5",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in output:
            row = {k: r.get(k) for k in columns}
            row["top5"] = json.dumps(r["top5"], ensure_ascii=False)
            writer.writerow(row)

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print(f"{'METHOD':10s} {'R@5':>9s} {'R@10':>9s} {'MRR':>9s} {'Rank1':>8s} {'Top5':>8s}")

    for method, m in aggregate.items():
        print(
            f"{method:10s} "
            f"{m['recall@5']:9.4f} "
            f"{m['recall@10']:9.4f} "
            f"{m['mrr']:9.4f} "
            f"{m['gold_rank_1_count']:8d} "
            f"{m['gold_rank_5_count']:8d}"
        )

    print("\nSaved:")
    print(json_path)
    print(csv_path)

if __name__ == "__main__":
    main_run()
