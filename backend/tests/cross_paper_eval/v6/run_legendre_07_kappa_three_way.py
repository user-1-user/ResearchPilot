import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import math
from pathlib import Path
from datetime import datetime

import sys

BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND_DIR))

import torch
import main

BASE = Path("tests/cross_paper_eval/v6")
RESULT = BASE / "results"

QID = "LEGENDRE_07"
GOLD_ID = "ee2443d1e7b24433a4c8a946ff76f744"

# ????????????
BAD_EN = "What is the contact form ? defined in the paper?"

# ????????
translated = json.loads(
    (RESULT / "language_generalization_sample80_corrected.json")
    .read_text(encoding="utf-8-sig")
)
item = next(x for x in translated if x["question_id"] == QID)

QUERIES = {
    "chinese_original": None,
    "english_original_bad": BAD_EN,
    "english_corrected": item["en_question"],
}

if "\u03ba" not in QUERIES["english_corrected"]:
    raise RuntimeError("Corrected English query has lost Greek kappa")

# ???? Stage1 ?????????
stage1 = json.loads(
    (RESULT / "cross_paper_stage1_retrieval.json")
    .read_text(encoding="utf-8-sig")
)
record = stage1["results"][QID]
QUERIES["chinese_original"] = record["question"]
candidate_ids = record["rrf_candidate_ids"]

if GOLD_ID not in candidate_ids:
    raise RuntimeError("Gold chunk is missing from Stage1 candidates")

if record["question"] != QUERIES["chinese_original"]:
    raise RuntimeError("Chinese query differs from original Stage1 query")

# ???? V6 Chunk ??
chunk_path = BASE / "v6_chunks.jsonl"
chunk_map = {}
with chunk_path.open(encoding="utf-8-sig") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            chunk_map[row["id"]] = row["text"]

missing = [cid for cid in candidate_ids if cid not in chunk_map]
if missing:
    raise RuntimeError(f"Missing candidate chunks: {missing[:5]}")

pairs_by_query = {
    name: [[query, chunk_map[cid][:1800]] for cid in candidate_ids]
    for name, query in QUERIES.items()
}

print("Question:", QID)
print("Candidates:", len(candidate_ids))
print("Gold candidate position:", candidate_ids.index(GOLD_ID) + 1)
print("Model:", main.RERANKER_MODEL)
print("Loading original V6 reranker...")

reranker = main._get_reranker()
print("Device:", getattr(reranker, "device", "unknown"))

output = {
    "experiment": "LEGENDRE_07 three-way query-language reranking",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "question_id": QID,
    "gold_chunk_id": GOLD_ID,
    "candidate_count": len(candidate_ids),
    "candidate_ids": candidate_ids,
    "model": str(main.RERANKER_MODEL),
    "max_pair_chars": 1800,
    "batch_size": 1,
    "activation": "torch.nn.Identity",
    "queries": {},
}

for name, query in QUERIES.items():
    print(f"\nRunning: {name}")
    pairs = pairs_by_query[name]

    scores = reranker.predict(
        pairs,
        batch_size=1,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
    )

    if len(scores) != len(candidate_ids):
        raise RuntimeError(f"{name}: score count mismatch")

    score_pairs = []
    for cid, score in zip(candidate_ids, scores):
        value = float(score)
        if not math.isfinite(value):
            raise RuntimeError(f"{name}: non-finite score")
        score_pairs.append((cid, value))

    ranking = [
        cid for cid, score in sorted(
            score_pairs, key=lambda x: x[1], reverse=True
        )
    ]

    gold_rank = ranking.index(GOLD_ID) + 1
    gold_score = dict(score_pairs)[GOLD_ID]

    output["queries"][name] = {
        "query": query,
        "ranking": ranking,
        "scores": {cid: score for cid, score in score_pairs},
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "mrr": 1.0 / gold_rank,
        "top1_is_gold": ranking[0] == GOLD_ID,
        "top10_contains_gold": GOLD_ID in ranking[:10],
    }

    print("Gold rank:", gold_rank)
    print("Gold score:", gold_score)
    print("Top 1:", ranking[0])
    print("Top 10 contains Gold:", GOLD_ID in ranking[:10])

# ??????
RESULT.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = RESULT / f"legendre_07_kappa_three_way_{timestamp}.json"
csv_path = RESULT / f"legendre_07_kappa_three_way_{timestamp}.csv"

json_path.write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query_type", "query", "gold_rank", "gold_score",
            "mrr", "top1_is_gold", "top10_contains_gold",
        ],
    )
    writer.writeheader()
    for name, result in output["queries"].items():
        writer.writerow({
            "query_type": name,
            "query": result["query"],
            "gold_rank": result["gold_rank"],
            "gold_score": result["gold_score"],
            "mrr": result["mrr"],
            "top1_is_gold": result["top1_is_gold"],
            "top10_contains_gold": result["top10_contains_gold"],
        })

print("\n========== COMPARISON ==========")
for name, result in output["queries"].items():
    print(
        f'{name}: rank={result["gold_rank"]}, '
        f'score={result["gold_score"]:.6f}, '
        f'MRR={result["mrr"]:.6f}'
    )

print("\nJSON:", json_path)
print("CSV:", csv_path)
print("Original Stage1/Stage2 files were not modified.")
