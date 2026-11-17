import sys
import json
import math
import gc
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main

EVAL_PATH = ROOT / "tests" / "evaluate_retrieval_lowmem_v2.py"
spec = importlib.util.spec_from_file_location("historical_eval", EVAL_PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OUT_DIR = ROOT / "tests" / "retrieval_eval_results" / "experimental_28_vs_38"
STAGE1 = OUT_DIR / "stage1_candidates_38.json"
STAGE2 = OUT_DIR / "stage2_rerank_comparison.json"

TOP28 = 28
TOP38 = 38


def save_new(path, data):
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment file: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def phase_retrieve():
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if STAGE1.exists():
        raise FileExistsError(f"Already exists: {STAGE1}")

    dataset = ev.load_dataset()
    questions = dataset["questions"]
    paper, rows, all_ids = ev.load_paper()

    for q in questions:
        missing = set(q["relevant_chunk_ids"]) - all_ids
        if missing:
            raise RuntimeError(f"{q['id']} missing Gold chunks: {missing}")

    if not main.LOCAL_EMBEDDING_MODEL:
        raise RuntimeError("LOCAL_EMBEDDING_MODEL is not configured.")

    if main.embedding_client:
        raise RuntimeError(
            "Embedding API is configured. This experiment requires local BGE-M3."
        )

    model = main._get_local_embedding_model()
    if model is None:
        raise RuntimeError("Could not load local BGE-M3.")

    docs = [row["text"] for row in rows]
    results = {}

    print("=" * 65)
    print("PHASE 1: REBUILD 38 RRF CANDIDATES")
    print("Paper:", paper["filename"])
    print("Questions:", len(questions))
    print("Chunks:", len(rows))
    print("BGE-M3:", main.LOCAL_EMBEDDING_MODEL)
    print("Threshold:", ev.DENSE_MIN_SCORE)
    print("=" * 65)

    for position, q in enumerate(questions, 1):
        qid = q["id"]

        encoded = model.encode(
            [q["question"]],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(encoded) != 1:
            raise RuntimeError(f"Embedding failed: {qid}")

        query_vector = encoded[0].astype(float).tolist()
        del encoded

        if len(query_vector) != ev.VECTOR_DIM:
            raise RuntimeError(f"Embedding dimension error: {qid}")

        sparse = main._bm25_scores(q["question"], docs)
        dense = [
            main._cosine(
                query_vector,
                json.loads(row["embedding_json"]),
            )
            for row in rows
        ]
        has_dense = bool(query_vector and any(dense))

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

        # Match historical evaluator's tie-breaking behavior.
        fused.sort(reverse=True)

        eligible = {
            i for i in range(len(rows))
            if sparse[i] > 0
            or (has_dense and dense[i] >= ev.DENSE_MIN_SCORE)
        }

        full_ids = [
            rows[i]["id"]
            for score, i in fused
            if i in eligible
        ][:TOP38]

        gold = set(q["relevant_chunk_ids"])
        if not gold.issubset(all_ids):
            raise RuntimeError(f"Gold validation failed: {qid}")

        results[qid] = {
            "rrf_candidate_ids": full_ids,
            "gold_ids": q["relevant_chunk_ids"],
            "question": q["question"],
            "diagnostic": {
                "eligible_count": len(eligible),
                "candidate_count": len(full_ids),
            },
        }

        print(
            f"[{position}/{len(questions)}] {qid}: "
            f"candidates={len(full_ids)}, "
            f"Gold in 28={len(set(full_ids[:TOP28]) & gold)}/{len(gold)}, "
            f"Gold in 38={len(set(full_ids) & gold)}/{len(gold)}",
            flush=True,
        )

        del query_vector, sparse, dense, fused
        gc.collect()

    state = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "paper_id": ev.PAPER_ID,
        "question_ids": [q["id"] for q in questions],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_limit": TOP38,
        "dense_min_score": ev.DENSE_MIN_SCORE,
        "rrf_k": 60,
        "results": results,
    }
    save_new(STAGE1, state)

    print("\nPHASE 1 PASS")
    print("Saved:", STAGE1)
    print("BGE-M3 process will exit now.")


def calculate_metrics(ranking, gold_ids):
    gold = set(gold_ids)
    if not gold:
        raise ValueError("Gold set is empty.")

    recall5 = len(set(ranking[:5]) & gold) / len(gold)
    recall10 = len(set(ranking[:10]) & gold) / len(gold)

    first_rank = next(
        (rank for rank, cid in enumerate(ranking, 1) if cid in gold),
        None,
    )
    mrr = 1 / first_rank if first_rank else 0.0

    return {
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "mrr": mrr,
        "first_relevant_rank": first_rank,
    }


def phase_rerank():
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if STAGE2.exists():
        raise FileExistsError(f"Refusing to overwrite: {STAGE2}")
    if not STAGE1.exists():
        raise FileNotFoundError(f"Run retrieve first: {STAGE1}")

    dataset = ev.load_dataset()
    state1 = load_json(STAGE1)
    questions = dataset["questions"]

    if state1["question_ids"] != [q["id"] for q in questions]:
        raise RuntimeError("Question ID mismatch.")
    if state1["paper_id"] != ev.PAPER_ID:
        raise RuntimeError("Paper ID mismatch.")

    paper, rows, all_ids = ev.load_paper()
    text_by_id = {row["id"]: row["text"] for row in rows}

    for q in questions:
        if q["id"] not in state1["results"]:
            raise RuntimeError(f"Missing candidates: {q['id']}")

    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("CrossEncoder failed to load.")

    print("=" * 65)
    print("PHASE 2: SAME CROSSENCODER, 28 VS 38")
    print("Paper:", paper["filename"])
    print("Device:", getattr(reranker, "device", "unknown"))
    print("Batch size: 1")
    print("Text length: 1800")
    print("=" * 65)

    per_question = {}

    for position, q in enumerate(questions, 1):
        qid = q["id"]
        candidate_ids = state1["results"][qid]["rrf_candidate_ids"]

        if len(candidate_ids) != TOP38:
            raise RuntimeError(
                f"{qid}: expected 38 candidates, got {len(candidate_ids)}"
            )

        if len(set(candidate_ids)) != TOP38:
            raise RuntimeError(f"Duplicate candidate IDs: {qid}")

        for cid in candidate_ids:
            if cid not in text_by_id:
                raise RuntimeError(f"Missing candidate text: {cid}")

        pairs = [
            [q["question"], text_by_id[cid][:1800]]
            for cid in candidate_ids
        ]

        scores = reranker.predict(
            pairs,
            batch_size=1,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )

        if len(scores) != TOP38:
            raise RuntimeError(f"Score count mismatch: {qid}")

        scored38 = [
            (float(score), cid)
            for score, cid in zip(scores, candidate_ids)
        ]
        if not all(math.isfinite(score) for score, cid in scored38):
            raise RuntimeError(f"Non-finite score: {qid}")

        # Same scores, different candidate pool.
        scored38.sort(key=lambda x: x[0], reverse=True)
        allowed28 = set(candidate_ids[:TOP28])
        scored28 = [item for item in scored38 if item[1] in allowed28]

        ranking28 = [cid for score, cid in scored28]
        ranking38 = [cid for score, cid in scored38]
        gold = q["relevant_chunk_ids"]

        per_question[qid] = {
            "gold_count": len(gold),
            "metrics_28": calculate_metrics(ranking28, gold),
            "metrics_38": calculate_metrics(ranking38, gold),
            "ranking_28": ranking28,
            "ranking_38": ranking38,
            "new_gold_in_38": list(
                (set(ranking38) & set(gold))
                - (set(ranking28) & set(gold))
            ),
        }

        print(
            f"[{position}/{len(questions)}] {qid}: "
            f"MRR28={per_question[qid]['metrics_28']['mrr']:.4f}, "
            f"MRR38={per_question[qid]['metrics_38']['mrr']:.4f}, "
            f"first28={per_question[qid]['metrics_28']['first_relevant_rank']}, "
            f"first38={per_question[qid]['metrics_38']['first_relevant_rank']}",
            flush=True,
        )

        del pairs, scores, scored38, scored28
        gc.collect()

    def macro(key, metric):
        return round(
            mean(row[key][metric] for row in per_question.values()),
            6,
        )

    gold_total = sum(len(q["relevant_chunk_ids"]) for q in questions)
    covered28 = sum(
        len(set(state1["results"][q["id"]]["rrf_candidate_ids"][:TOP28])
            & set(q["relevant_chunk_ids"]))
        for q in questions
    )
    covered38 = sum(
        len(set(state1["results"][q["id"]]["rrf_candidate_ids"])
            & set(q["relevant_chunk_ids"]))
        for q in questions
    )

    summary = {
        "question_count": len(questions),
        "gold_total": gold_total,
        "candidate_gold_coverage_28": covered28,
        "candidate_gold_coverage_38": covered38,
        "reranked_28": {
            "macro_recall_at_5": macro("metrics_28", "recall_at_5"),
            "macro_recall_at_10": macro("metrics_28", "recall_at_10"),
            "macro_mrr": macro("metrics_28", "mrr"),
        },
        "reranked_38": {
            "macro_recall_at_5": macro("metrics_38", "recall_at_5"),
            "macro_recall_at_10": macro("metrics_38", "recall_at_10"),
            "macro_mrr": macro("metrics_38", "mrr"),
        },
    }

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "Same CrossEncoder, nested candidate pools 28 vs 38",
        "model_path": main.RERANKER_MODEL,
        "batch_size": 1,
        "text_truncation": 1800,
        "activation": "torch.nn.Identity",
        "summary": summary,
        "per_question": per_question,
    }
    save_new(STAGE2, report)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nQ02:")
    print(json.dumps(per_question.get("Q02", {}), ensure_ascii=False, indent=2))
    print("\nPHASE 2 PASS")
    print("Saved:", STAGE2)
    print("No historical files or database records were modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["retrieve", "rerank"],
        required=True,
    )
    args = parser.parse_args()

    if args.phase == "retrieve":
        phase_retrieve()
    else:
        phase_rerank()
