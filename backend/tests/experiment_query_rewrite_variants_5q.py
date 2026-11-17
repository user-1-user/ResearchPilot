import sys
import json
import csv
import math
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

VARIANTS = {
    "Q08": [
        "What computation does the position-wise feed-forward network perform independently at each position?",
        "How is the feed-forward sub-layer applied to each position in the Transformer?",
        "What are the operations performed by the position-wise feed-forward network in the Transformer?",
    ],
    "Q29": [
        "What scaling factor is applied to the embedding weights in the original Transformer?",
        "How are the weights of the source and target embedding layers scaled?",
        "What is the weight scaling method used for the Transformer embedding layers?",
    ],
    "Q31": [
        "In which three ways is multi-head attention applied in the Transformer?",
        "How does the Transformer use multi-head attention in its encoder and decoder?",
        "What are the three attention applications described for the Transformer architecture?",
    ],
    "Q42": [
        "What beam width and length penalty alpha were used for machine translation?",
        "What decoding parameters did the Transformer use for machine translation inference?",
        "What beam search size and length normalization setting were used during translation?",
    ],
    "Q49": [
        "Which directions for future work on the Transformer are identified in the paper?",
        "What further research does the paper suggest following the introduction of the Transformer?",
        "What future extensions or research possibilities do the authors discuss for the Transformer?",
    ],
}

def rank_details(scored, gold_ids):
    gold = set(gold_ids)
    ordered = sorted(scored, key=lambda x: x[0], reverse=True)

    first_gold_rank = None
    best_gold_score = None
    best_gold_id = None
    best_negative_score = None
    best_negative_id = None

    for rank, (score, cid) in enumerate(ordered, 1):
        if cid in gold:
            if first_gold_rank is None:
                first_gold_rank = rank
            if best_gold_score is None or score > best_gold_score:
                best_gold_score = score
                best_gold_id = cid
        else:
            if best_negative_score is None or score > best_negative_score:
                best_negative_score = score
                best_negative_id = cid

    margin = None
    if best_gold_score is not None and best_negative_score is not None:
        margin = round(best_gold_score - best_negative_score, 6)

    return {
        "first_gold_rank": first_gold_rank,
        "best_gold_score": round(best_gold_score, 6) if best_gold_score is not None else None,
        "best_gold_id": best_gold_id,
        "best_non_gold_score": round(best_negative_score, 6) if best_negative_score is not None else None,
        "best_non_gold_id": best_negative_id,
        "gold_vs_negative_margin": margin,
        "top5": [
            {
                "rank": i,
                "chunk_id": cid,
                "score": round(score, 6),
                "is_gold": cid in gold,
            }
            for i, (score, cid) in enumerate(ordered[:5], 1)
        ],
    }


def main_run():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source result: {SOURCE}")

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_rows = {r["id"]: r for r in source["results"]}

    for qid in VARIANTS:
        if qid not in source_rows:
            raise RuntimeError(f"Missing question in source: {qid}")

    # Load chunk text from the original evaluation module.
    import importlib.util
    eval_path = ROOT / "tests/evaluate_retrieval_lowmem_v2.py"
    spec = importlib.util.spec_from_file_location("eval_module", eval_path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    paper, chunks, all_ids = ev.load_paper()
    text_by_id = {r["id"]: r["text"] for r in chunks}

    for qid, r in source_rows.items():
        if qid not in VARIANTS:
            continue
        for cid in r["original_rankings"]["RRF"]:
            if cid not in text_by_id:
                raise RuntimeError(f"Missing candidate text: {cid}")

    print("Paper:", paper["filename"])
    print("Chunks:", len(chunks))
    print("Loading CrossEncoder...")

    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("CrossEncoder failed to load")

    output = []

    for qid, variants in VARIANTS.items():
        r = source_rows[qid]
        candidates = r["original_rankings"]["RRF"]
        gold_ids = r["gold_ids"]
        original_query = r["original_question"]

        if len(candidates) != 28:
            raise RuntimeError(
                f"{qid}: expected 28 fixed candidates, got {len(candidates)}"
            )

        tests = [("ORIGINAL", original_query)]
        tests.extend(
            (f"VARIANT_{i}", query)
            for i, query in enumerate(variants, 1)
        )

        print("\n" + "=" * 70)
        print(qid, "| fixed candidates:", len(candidates))

        for label, query in tests:
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

            if len(scores) != len(candidates):
                raise RuntimeError(f"{qid} {label}: score count mismatch")

            scored = [
                (float(score), cid)
                for score, cid in zip(scores, candidates)
            ]

            if not all(math.isfinite(score) for score, cid in scored):
                raise RuntimeError(f"{qid} {label}: invalid score")

            details = rank_details(scored, gold_ids)

            record = {
                "question_id": qid,
                "label": label,
                "query": query,
                "candidate_count": len(candidates),
                "candidate_ids": candidates,
                **details,
            }
            output.append(record)

            print(
                f"{label:<12} Gold rank={str(details['first_gold_rank']):>3}"
                f" | Gold score={str(details['best_gold_score']):>10}"
                f" | Best negative={str(details['best_non_gold_score']):>10}"
                f" | Margin={str(details['gold_vs_negative_margin']):>10}"
            )

            for item in details["top5"]:
                mark = "GOLD" if item["is_gold"] else ""
                print(
                    f"  {item['rank']}. {item['chunk_id']} "
                    f"score={item['score']} {mark}"
                )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / f"query_rewrite_variants_5q_{stamp}.json"
    csv_path = OUT_DIR / f"query_rewrite_variants_5q_{stamp}.csv"

    report = {
        "experiment": "Five-question fixed-pool query rewrite variants",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_result": SOURCE.name,
        "candidate_policy": "Original RRF candidate pool fixed at 28 per question",
        "reranker_settings": {
            "text_limit": 1800,
            "batch_size": 1,
            "activation": "Identity",
        },
        "results": output,
    }

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "label", "query", "candidate_count",
            "first_gold_rank", "best_gold_score",
            "best_non_gold_score", "gold_vs_negative_margin",
        ])
        for x in output:
            writer.writerow([
                x["question_id"], x["label"], x["query"],
                x["candidate_count"], x["first_gold_rank"],
                x["best_gold_score"], x["best_non_gold_score"],
                x["gold_vs_negative_margin"],
            ])

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main_run()
