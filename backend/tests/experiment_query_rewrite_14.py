import sys
import json
import csv
import math
import importlib.util
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datetime import datetime
from statistics import mean

import torch
import main

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Import the existing evaluation module without changing it.
spec = importlib.util.spec_from_file_location(
    "retrieval_eval",
    Path("tests/evaluate_retrieval_lowmem_v2.py"),
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

# One controlled rewrite per degraded question.
REWRITES = {
    "Q03": "What are the respective functions of the Transformer encoder and decoder?",
    "Q13": "What are the respective functions of the Transformer encoder and decoder?",
    "Q08": "How does the position-wise feed-forward network operate at each position in the Transformer?",
    "Q09": "What role does positional encoding play in the Transformer, and why is it needed?",
    "Q29": "How does the original Transformer scale the weights in its embedding layers?",
    "Q30": "How many attention heads does the Transformer use, and what are the key and value dimensions of each head?",
    "Q31": "What are the three different ways the Transformer uses multi-head attention?",
    "Q32": "How does the paper compare the computational complexity of self-attention, convolutional layers, and restricted self-attention?",
    "Q33": "How does the neighborhood size r in restricted self-attention affect the maximum path length?",
    "Q36": "What difference did the authors observe between learned positional embeddings and sinusoidal positional encodings?",
    "Q37": "Approximately how many source and target tokens are included in each training batch?",
    "Q42": "What beam size and length penalty were used during machine translation decoding?",
    "Q45": "How did changing the number of attention heads and key dimensions affect model performance?",
    "Q49": "What future research directions for the Transformer do the authors propose?",
}

TARGET_IDS = list(REWRITES)

def retrieve(query, rows, embedding_model):
    encoded = embedding_model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if len(encoded) != 1:
        raise RuntimeError("Query embedding failed")

    query_vector = encoded[0].astype(float).tolist()
    del encoded

    if len(query_vector) != ev.VECTOR_DIM:
        raise RuntimeError("Query vector dimension mismatch")

    if not all(math.isfinite(float(x)) for x in query_vector):
        raise RuntimeError("Invalid query vector")

    docs = [row["text"] for row in rows]
    sparse = main._bm25_scores(query, docs)

    dense = [
        main._cosine(
            query_vector,
            json.loads(row["embedding_json"]),
        )
        for row in rows
    ]

    has_dense = bool(query_vector and any(dense))

    bm25_indices = sorted(
        [i for i in range(len(rows)) if sparse[i] > 0],
        key=lambda i: sparse[i],
        reverse=True,
    )

    dense_indices = (
        sorted(
            range(len(rows)),
            key=lambda i: dense[i],
            reverse=True,
        )
        if has_dense else []
    )

    sparse_rank = {
        idx: rank
        for rank, idx in enumerate(
            sorted(
                range(len(rows)),
                key=lambda i: sparse[i],
                reverse=True,
            ),
            1,
        )
    }

    dense_rank = {
        idx: rank
        for rank, idx in enumerate(
            sorted(
                range(len(rows)),
                key=lambda i: dense[i],
                reverse=True,
            ),
            1,
        )
    }

    fused = []
    for i in range(len(rows)):
        score = 1 / (60 + sparse_rank[i])
        if has_dense:
            score += 1 / (60 + dense_rank[i])
        fused.append((score, i))

    fused.sort(reverse=True)

    eligible = {
        i for i in range(len(rows))
        if sparse[i] > 0
        or (has_dense and dense[i] >= ev.DENSE_MIN_SCORE)
    }

    candidate_limit = max(20, min(ev.EVAL_TOP_K * 4, 40))

    rrf_indices = [
        idx for score, idx in fused if idx in eligible
    ][:candidate_limit]

    return {
        "BM25": [rows[i]["id"] for i in bm25_indices],
        "Dense_BGE_M3": [rows[i]["id"] for i in dense_indices],
        "RRF": [rows[i]["id"] for i in rrf_indices],
    }


def rerank(query, candidate_ids, text_by_id, model):
    if not candidate_ids:
        return []

    pairs = [
        [query, text_by_id[cid][:1800]]
        for cid in candidate_ids
    ]

    scores = model.predict(
        pairs,
        batch_size=1,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
    )

    if len(scores) != len(candidate_ids):
        raise RuntimeError("Reranker output mismatch")

    scored = [
        (float(score), cid)
        for score, cid in zip(scores, candidate_ids)
    ]

    if not all(math.isfinite(score) for score, cid in scored):
        raise RuntimeError("Invalid reranker score")

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for score, cid in scored]


def metric(ranking, gold_ids):
    return ev.calculate_metrics(ranking, gold_ids)


def summarize(rows):
    result = {}
    for method in rows[0]["metrics"]:
        values = [r["metrics"][method] for r in rows]
        result[method] = {
            "question_count": len(values),
            "recall_at_5": round(
                mean(x["recall_at_5"] for x in values), 4
            ),
            "recall_at_10": round(
                mean(x["recall_at_10"] for x in values), 4
            ),
            "mrr": round(
                mean(x["reciprocal_rank"] for x in values), 4
            ),
        }
    return result


def main_run():
    dataset = ev.load_dataset()
    questions = {
        q["id"]: q
        for q in dataset["questions"]
        if q["id"] in TARGET_IDS
    }

    missing = set(TARGET_IDS) - set(questions)
    if missing:
        raise RuntimeError(f"Questions missing from dataset: {missing}")

    paper, rows, all_ids = ev.load_paper()
    text_by_id = {r["id"]: r["text"] for r in rows}

    for qid, q in questions.items():
        if set(q["relevant_chunk_ids"]) - all_ids:
            raise RuntimeError(f"Gold chunks missing: {qid}")

    print("Paper:", paper["filename"])
    print("Chunks:", len(rows))
    print("Target questions:", len(TARGET_IDS))
    print("Loading local BGE-M3...")

    embedding_model = main._get_local_embedding_model()
    if embedding_model is None:
        raise RuntimeError("BGE-M3 failed to load")

    print("Loading CrossEncoder...")
    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("Reranker failed to load")

    output = []
    print("\nStarting original vs rewritten evaluation...\n")

    for position, qid in enumerate(TARGET_IDS, 1):
        q = questions[qid]
        original_query = q["question"]
        rewritten_query = REWRITES[qid]
        gold_ids = q["relevant_chunk_ids"]

        print(
            f"[{position}/{len(TARGET_IDS)}] {qid}",
            flush=True,
        )

        # Original full retrieval pipeline.
        original_rankings = retrieve(
            original_query, rows, embedding_model
        )
        original_candidates = original_rankings["RRF"]

        original_reranked = rerank(
            original_query,
            original_candidates,
            text_by_id,
            reranker,
        )

        # Rewritten full retrieval pipeline.
        rewritten_rankings = retrieve(
            rewritten_query, rows, embedding_model
        )
        rewritten_candidates = rewritten_rankings["RRF"]

        rewritten_reranked = rerank(
            rewritten_query,
            rewritten_candidates,
            text_by_id,
            reranker,
        )

        # Controlled test: same original candidate pool,
        # only change the CrossEncoder query.
        rewritten_fixed_pool = rerank(
            rewritten_query,
            original_candidates,
            text_by_id,
            reranker,
        )

        original_metrics = {
            "BM25": metric(original_rankings["BM25"], gold_ids),
            "Dense_BGE_M3": metric(
                original_rankings["Dense_BGE_M3"], gold_ids
            ),
            "RRF": metric(original_rankings["RRF"], gold_ids),
            "RRF_plus_BGE_Reranker": metric(
                original_reranked, gold_ids
            ),
        }

        rewritten_metrics = {
            "BM25": metric(rewritten_rankings["BM25"], gold_ids),
            "Dense_BGE_M3": metric(
                rewritten_rankings["Dense_BGE_M3"], gold_ids
            ),
            "RRF": metric(rewritten_rankings["RRF"], gold_ids),
            "RRF_plus_BGE_Reranker": metric(
                rewritten_reranked, gold_ids
            ),
            "Reranker_fixed_original_pool": metric(
                rewritten_fixed_pool, gold_ids
            ),
        }

        record = {
            "id": qid,
            "topic": q["topic"],
            "original_question": original_query,
            "rewritten_question": rewritten_query,
            "gold_ids": gold_ids,
            "original_candidate_count": len(original_candidates),
            "rewritten_candidate_count": len(rewritten_candidates),
            "original_rankings": {
                **original_rankings,
                "RRF_plus_BGE_Reranker": original_reranked,
            },
            "rewritten_rankings": {
                **rewritten_rankings,
                "RRF_plus_BGE_Reranker": rewritten_reranked,
                "Reranker_fixed_original_pool": rewritten_fixed_pool,
            },
            "original_metrics": original_metrics,
            "rewritten_metrics": rewritten_metrics,
        }
        output.append(record)

        old_rank = original_metrics[
            "RRF_plus_BGE_Reranker"
        ]["first_relevant_rank"]
        new_rank = rewritten_metrics[
            "RRF_plus_BGE_Reranker"
        ]["first_relevant_rank"]
        fixed_rank = rewritten_metrics[
            "Reranker_fixed_original_pool"
        ]["first_relevant_rank"]

        print(
            f"  Reranker rank: {old_rank} -> {new_rank}"
            f" | fixed pool: {fixed_rank}",
            flush=True,
        )

    original_summary = summarize([
        {
            "metrics": r["original_metrics"]
        }
        for r in output
    ])

    rewritten_summary = summarize([
        {
            "metrics": {
                k: v
                for k, v in r["rewritten_metrics"].items()
                if k != "Reranker_fixed_original_pool"
            }
        }
        for r in output
    ])

    fixed_summary = summarize([
        {
            "metrics": {
                "Reranker_fixed_original_pool":
                    r["rewritten_metrics"]["Reranker_fixed_original_pool"]
            }
        }
        for r in output
    ])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("tests/retrieval_eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"query_rewrite_14_{stamp}.json"
    csv_path = out_dir / f"query_rewrite_14_{stamp}.csv"

    report = {
        "experiment": "Controlled query rewrite experiment",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paper_id": ev.PAPER_ID,
        "chunk_count": len(rows),
        "question_ids": TARGET_IDS,
        "rewrite_policy": "One predefined rewrite per degraded question",
        "original_summary": original_summary,
        "rewritten_summary": rewritten_summary,
        "fixed_original_pool_summary": fixed_summary,
        "results": output,
    }

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "qid", "topic", "original_question", "rewritten_question",
            "original_reranker_rank", "rewritten_reranker_rank",
            "rewritten_fixed_pool_rank",
            "original_rrf_rank", "rewritten_rrf_rank",
            "original_recall5", "rewritten_recall5",
            "original_mrr", "rewritten_mrr",
        ])

        for r in output:
            om = r["original_metrics"]
            nm = r["rewritten_metrics"]

            writer.writerow([
                r["id"],
                r["topic"],
                r["original_question"],
                r["rewritten_question"],
                om["RRF_plus_BGE_Reranker"]["first_relevant_rank"],
                nm["RRF_plus_BGE_Reranker"]["first_relevant_rank"],
                nm["Reranker_fixed_original_pool"]["first_relevant_rank"],
                om["RRF"]["first_relevant_rank"],
                nm["RRF"]["first_relevant_rank"],
                om["RRF_plus_BGE_Reranker"]["recall_at_5"],
                nm["RRF_plus_BGE_Reranker"]["recall_at_5"],
                om["RRF_plus_BGE_Reranker"]["reciprocal_rank"],
                nm["RRF_plus_BGE_Reranker"]["reciprocal_rank"],
            ])

    print("\n" + "=" * 75)
    print("ORIGINAL SUMMARY")
    print(json.dumps(original_summary, indent=2))
    print("\nREWRITTEN SUMMARY")
    print(json.dumps(rewritten_summary, indent=2))
    print("\nFIXED ORIGINAL CANDIDATE POOL")
    print(json.dumps(fixed_summary, indent=2))
    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main_run()
