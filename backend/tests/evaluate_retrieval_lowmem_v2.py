from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os

# Limit CPU threads before importing PyTorch-dependent modules.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main


DATASET_PATH = BACKEND_DIR / "tests" / "retrieval_eval_dataset_v2_en.json"
RESULT_DIR = BACKEND_DIR / "tests" / "retrieval_eval_results"

STAGE1_PATH = RESULT_DIR / f"stage1_retrieval_{DATASET_PATH.stem}.json"
STAGE2_PATH = RESULT_DIR / f"stage2_reranking_{DATASET_PATH.stem}.json"

PAPER_ID = "11f4d8162178"
EXPECTED_CHUNKS = 38
VECTOR_DIM = 1024
EVAL_TOP_K = int(main.DEFAULT_TOP_K)
DENSE_MIN_SCORE = 0.18

METHODS = [
    "BM25",
    "Dense_BGE_M3",
    "RRF",
    "RRF_plus_BGE_Reranker",
]


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))

    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    os.replace(temp_path, path)


def load_dataset(dataset_path: Path | None = None) -> dict:
    dataset_path = dataset_path or DATASET_PATH
    dataset = load_json(dataset_path)

    if dataset["paper"]["paper_id"] != PAPER_ID:
        raise ValueError("Evaluation paper ID mismatch.")

    questions = dataset.get("questions", [])

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    ids = [q["id"] for q in questions]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs.")

    for q in questions:
        for field in ("id", "language", "topic", "question", "relevant_chunk_ids"):
            if field not in q:
                raise ValueError(f"Question {q.get('id')} missing field: {field}")
        if not isinstance(q["relevant_chunk_ids"], list):
            raise ValueError(f"Invalid relevant_chunk_ids in {q['id']}")
        if not q["relevant_chunk_ids"]:
            raise ValueError(f"Question {q['id']} has no gold chunks.")

    print("Dataset loaded:", dataset_path.name)
    print("Question count:", len(questions))
    return dataset


def validate_checkpoint(state: dict, dataset: dict) -> None:
    if state.get("dataset_name") != dataset["dataset_name"]:
        raise ValueError("Checkpoint dataset name mismatch.")

    if state.get("dataset_version") != dataset["version"]:
        raise ValueError("Checkpoint dataset version mismatch.")

    if state.get("paper_id") != PAPER_ID:
        raise ValueError("Checkpoint paper ID mismatch.")

    expected_ids = [q["id"] for q in dataset["questions"]]
    if state.get("question_ids") != expected_ids:
        raise ValueError("Checkpoint question ID list mismatch.")

    for field in (
        "query_language",
        "query_translation_status",
        "source_dataset",
        "source_translation_file",
    ):
        if state.get(field) != dataset.get(field):
            raise ValueError(f"Checkpoint metadata mismatch: {field}")


def load_paper():
    with main.db() as conn:
        paper = conn.execute(
            """
            SELECT id, user_id, filename
            FROM papers
            WHERE id = ?
            """,
            (PAPER_ID,),
        ).fetchone()

    if paper is None:
        raise RuntimeError("Target paper not found.")

    rows = main._load_user_chunks(
        paper["user_id"],
        paper_ids=[PAPER_ID],
    )

    if len(rows) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"Expected 38 chunks, found {len(rows)}"
        )

    ids = {row["id"] for row in rows}

    for row in rows:
        raw = row.get("embedding_json")

        if not raw:
            raise RuntimeError(
                f"Missing vector: {row['id']}"
            )

        vector = json.loads(raw)

        if len(vector) != VECTOR_DIM:
            raise RuntimeError(
                f"Vector dimension error: {row['id']}"
            )

        if not all(
            math.isfinite(float(x)) for x in vector
        ):
            raise RuntimeError(
                f"Invalid vector: {row['id']}"
            )

    return paper, rows, ids


# --------------------------------------------------
# Phase 1: BM25 + Dense + RRF
# Only BGE-M3 is loaded in this process.
# --------------------------------------------------

def phase_retrieve(max_questions=None):
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = load_dataset()
    questions = dataset["questions"]

    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        questions = questions[:max_questions]

    paper, rows, all_ids = load_paper()

    for q in questions:
        missing = set(q["relevant_chunk_ids"]) - all_ids
        if missing:
            raise RuntimeError(
                f"Missing gold chunks in {q['id']}: {missing}"
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if STAGE1_PATH.exists():
        state = load_json(STAGE1_PATH)
        validate_checkpoint(state, dataset)
        results = state.get("results", {})
        print("Existing checkpoint found. Resuming.")
    else:
        results = {}

    state = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "paper_id": PAPER_ID,
        "question_ids": [q["id"] for q in dataset["questions"]],
        "query_language": dataset.get("query_language"),
        "query_translation_status": dataset.get("query_translation_status"),
        "source_dataset": dataset.get("source_dataset"),
        "source_translation_file": dataset.get("source_translation_file"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    print("=" * 65)
    print("PHASE 1: BM25 + DENSE + RRF")
    print("=" * 65)
    print("Paper:", paper["filename"])
    print("Chunks:", len(rows))
    print("Questions:", len(questions))
    print("Model:", "BGE-M3 only")
    print()

    if main.embedding_client:
        backend = "Configured embedding API"
    else:
        backend = str(main.LOCAL_EMBEDDING_MODEL)

    # Explicitly load the local BGE-M3 model.
    # This process does not load the CrossEncoder reranker.
    if not main.LOCAL_EMBEDDING_MODEL:
        raise RuntimeError(
            "LOCAL_EMBEDDING_MODEL is not configured."
        )

    model = main._get_local_embedding_model()

    if model is None:
        raise RuntimeError(
            "Local BGE-M3 model could not be loaded."
        )

    print("Embedding backend: local BGE-M3")
    print("Model loaded successfully.")
    print()

    for position, q in enumerate(questions, 1):
        qid = q["id"]

        if qid in results:
            print(
                f"[{position}/{len(questions)}] {qid}: already completed, skip",
                flush=True,
            )
            continue

        print(
            f"[{position}/{len(questions)}] Encoding {qid}...",
            flush=True,
        )

        # Encode exactly one query with local BGE-M3.
        encoded = model.encode(
            [q["question"]],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if len(encoded) != 1:
            raise RuntimeError(
                f"Query embedding failed: {qid}"
            )

        query_vector = encoded[0].astype(float).tolist()
        del encoded

        if len(query_vector) != VECTOR_DIM:
            raise RuntimeError(
                f"Query vector dimension mismatch: {qid}"
            )

        if not all(
            math.isfinite(float(x)) for x in query_vector
        ):
            raise RuntimeError(
                f"Invalid query vector: {qid}"
            )

        docs = [row["text"] for row in rows]

        sparse = main._bm25_scores(q["question"], docs)

        dense = [
            main._cosine(
                query_vector,
                json.loads(row["embedding_json"]),
            )
            for row in rows
        ]

        has_dense = bool(query_vector and any(dense))

        # BM25 baseline: only documents with lexical matches.
        bm25_indices = sorted(
            [
                i for i in range(len(rows))
                if sparse[i] > 0
            ],
            key=lambda i: sparse[i],
            reverse=True,
        )

        # Dense baseline: rank all 38 chunks without
        # applying the production RRF threshold.
        dense_indices = (
            sorted(
                range(len(rows)),
                key=lambda i: dense[i],
                reverse=True,
            )
            if has_dense
            else []
        )

        # RRF: same rank fusion and eligibility rule
        # as the production retrieval implementation.
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
            or (
                has_dense
                and dense[i] >= DENSE_MIN_SCORE
            )
        }

        # top_k=10 gives a production-style candidate
        # pool of up to 40 chunks.
        candidate_limit = max(
            20,
            min(EVAL_TOP_K * 4, 40),
        )

        rrf_indices = [
            idx for score, idx in fused
            if idx in eligible
        ][:candidate_limit]

        rankings = {
            "BM25": [
                rows[i]["id"] for i in bm25_indices
            ],
            "Dense_BGE_M3": [
                rows[i]["id"] for i in dense_indices
            ],
            "RRF": [
                rows[i]["id"] for i in rrf_indices
            ],
        }

        results[qid] = {
            "question": q["question"],
            "language": q["language"],
            "topic": q["topic"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "rankings": rankings,
            "rrf_candidate_ids": rankings["RRF"],
            "diagnostic": {
                "bm25_candidates": len(bm25_indices),
                "dense_candidates": len(dense_indices),
                "rrf_candidates": len(rrf_indices),
                "has_dense": has_dense,
            },
        }

        # Save after every question for safe resumption.
        state["results"] = results
        save_json_atomic(STAGE1_PATH, state)

        print(
            f"  Saved {qid}: "
            f"BM25={len(bm25_indices)}, "
            f"Dense={len(dense_indices)}, "
            f"RRF={len(rrf_indices)}",
            flush=True,
        )

        # Discard query-specific objects before next question.
        del query_vector, sparse, dense, fused
        gc.collect()
        gc.collect()

    print()
    print("PHASE 1 COMPLETE")
    print("Checkpoint:", STAGE1_PATH)
    print("Completed questions:", len(results))
    print("BGE-M3 will be released when this process exits.")


# --------------------------------------------------
# Phase 2: CrossEncoder reranking
# No BGE-M3 query model is loaded in this process.
# --------------------------------------------------

def phase_rerank(max_questions=None):
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = load_dataset()
    questions = dataset["questions"]

    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        questions = questions[:max_questions]

    if not STAGE1_PATH.exists():
        raise FileNotFoundError(
            "Run --phase retrieve first."
        )

    stage1 = load_json(STAGE1_PATH)
    validate_checkpoint(stage1, dataset)

    stage1_results = stage1.get("results", {})

    missing = [
        q["id"] for q in questions
        if q["id"] not in stage1_results
    ]

    if missing:
        raise RuntimeError(
            "Phase 1 is incomplete. Missing: " + str(missing)
        )

    paper, rows, all_ids = load_paper()

    text_by_id = {
        row["id"]: row["text"]
        for row in rows
    }

    if STAGE2_PATH.exists():
        state = load_json(STAGE2_PATH)
        validate_checkpoint(state, dataset)
        results = state.get("results", {})
        print("Existing reranker checkpoint found. Resuming.")
    else:
        results = {}

    state = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "paper_id": PAPER_ID,
        "question_ids": [q["id"] for q in dataset["questions"]],
        "query_language": dataset.get("query_language"),
        "query_translation_status": dataset.get("query_translation_status"),
        "source_dataset": dataset.get("source_dataset"),
        "source_translation_file": dataset.get("source_translation_file"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    print("=" * 65)
    print("PHASE 2: BGE RERANKER")
    print("=" * 65)
    print("Paper:", paper["filename"])
    print("Model: CrossEncoder only")
    print("Batch size: 1")
    print()

    reranker = main._get_reranker()

    if reranker is None:
        raise RuntimeError("Reranker could not be loaded.")

    print("Reranker device:", getattr(reranker, "device", "unknown"))

    import torch

    for position, q in enumerate(questions, 1):
        qid = q["id"]

        if qid in results:
            print(
                f"[{position}/{len(questions)}] {qid}: already completed, skip",
                flush=True,
            )
            continue

        candidate_ids = stage1_results[qid]["rrf_candidate_ids"]

        for cid in candidate_ids:
            if cid not in text_by_id:
                raise RuntimeError(
                    f"Candidate chunk not found: {cid}"
                )

        print(
            f"[{position}/{len(questions)}] Reranking {qid}, "
            f"candidates={len(candidate_ids)}...",
            flush=True,
        )

        if candidate_ids:
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

            if len(scores) != len(candidate_ids):
                raise RuntimeError(
                    f"Reranker output mismatch: {qid}"
                )

            scored = [
                (float(score), cid)
                for score, cid in zip(scores, candidate_ids)
            ]

            if not all(
                math.isfinite(score)
                for score, cid in scored
            ):
                raise RuntimeError(
                    f"Invalid reranker score: {qid}"
                )

            scored.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            reranked_ids = [
                cid for score, cid in scored
            ]

            score_map = {
                cid: round(score, 6)
                for score, cid in scored
            }

        else:
            reranked_ids = []
            score_map = {}

        results[qid] = {
            "RRF_plus_BGE_Reranker": reranked_ids,
            "rerank_scores": score_map,
        }

        state["results"] = results
        save_json_atomic(STAGE2_PATH, state)

        print(
            f"  Saved {qid}: {len(reranked_ids)} reranked candidates",
            flush=True,
        )

        if candidate_ids:
            del pairs

    print()
    print("PHASE 2 COMPLETE")
    print("Checkpoint:", STAGE2_PATH)
    print("Completed questions:", len(results))
    print("Reranker will be released when this process exits.")


# --------------------------------------------------
# Metric calculation
# --------------------------------------------------

def calculate_metrics(ranking, gold_ids):
    gold = set(gold_ids)

    recall5 = len(set(ranking[:5]) & gold) / len(gold)
    recall10 = len(set(ranking[:10]) & gold) / len(gold)

    first_rank = None

    for rank, chunk_id in enumerate(ranking, 1):
        if chunk_id in gold:
            first_rank = rank
            break

    reciprocal_rank = (
        1 / first_rank if first_rank is not None else 0.0
    )

    return {
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_rank,
    }


def phase_summary():
    dataset = load_dataset()
    questions = dataset["questions"]

    stage1 = load_json(STAGE1_PATH)
    stage2 = load_json(STAGE2_PATH)

    validate_checkpoint(stage1, dataset)
    validate_checkpoint(stage2, dataset)

    results1 = stage1["results"]
    results2 = stage2["results"]

    question_results = []

    for q in questions:
        qid = q["id"]

        if qid not in results1 or qid not in results2:
            raise RuntimeError(
                f"Missing phase results for {qid}"
            )

        rankings = dict(results1[qid]["rankings"])
        rankings["RRF_plus_BGE_Reranker"] = (
            results2[qid]["RRF_plus_BGE_Reranker"]
        )

        metrics = {
            method: calculate_metrics(
                rankings[method],
                q["relevant_chunk_ids"],
            )
            for method in METHODS
        }

        question_results.append({
            "id": qid,
            "language": q["language"],
            "topic": q["topic"],
            "relevant_chunk_count": len(q["relevant_chunk_ids"]),
            "rankings": rankings,
            "metrics": metrics,
        })

    groups = {
        "all": question_results,
        "zh": [q for q in question_results if q["language"] == "zh"],
        "en": [q for q in question_results if q["language"] == "en"],
    }

    summary = {}

    for method in METHODS:
        summary[method] = {}

        for group_name, group_rows in groups.items():
            values = [q["metrics"][method] for q in group_rows]

            if not values:
                summary[method][group_name] = {
                    "question_count": 0,
                    "recall_at_5": None,
                    "recall_at_10": None,
                    "mrr": None,
                }
            else:
                summary[method][group_name] = {
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

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = RESULT_DIR / f"retrieval_eval_{stamp}.json"
    csv_path = RESULT_DIR / f"retrieval_eval_{stamp}.csv"

    report = {
        "report_name": "Low-memory RAG Retrieval Evaluation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "annotation_status": dataset["annotation_policy"]["status"],
        "query_language": dataset.get("query_language"),
        "query_translation_status": dataset.get("query_translation_status"),
        "source_dataset": dataset.get("source_dataset"),
        "source_translation_file": dataset.get("source_translation_file"),
        "paper_id": PAPER_ID,
        "question_count": len(questions),
        "chunk_count": EXPECTED_CHUNKS,
        "methods": METHODS,
        "metric_averaging": "Macro average across questions",
        "summary": summary,
        "per_question_results": question_results,
    }

    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Report filename collision.")

    save_json_atomic(json_path, report)

    csv_rows = []

    for q in question_results:
        for method in METHODS:
            metric = q["metrics"][method]

            csv_rows.append({
                "question_id": q["id"],
                "language": q["language"],
                "topic": q["topic"],
                "method": method,
                "recall_at_5": round(metric["recall_at_5"], 6),
                "recall_at_10": round(metric["recall_at_10"], 6),
                "mrr": round(metric["reciprocal_rank"], 6),
                "first_relevant_rank": metric["first_relevant_rank"],
                "top10_chunk_ids": json.dumps(
                    q["rankings"][method][:10]
                ),
            })

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(csv_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 85)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 85)

    for group_name in ("all", "zh", "en"):
        print()
        print("GROUP:", group_name.upper())
        print(
            f"{'Method':<28}"
            f"{'N':>5}"
            f"{'Recall@5':>12}"
            f"{'Recall@10':>12}"
            f"{'MRR':>10}"
        )

        for method in METHODS:
            item = summary[method][group_name]

            print(
                f"{method:<28}"
                f"{item['question_count']:>5}"
                f"{item['recall_at_5'] if item['recall_at_5'] is not None else 'N/A':>12}"
                f"{item['recall_at_10'] if item['recall_at_10'] is not None else 'N/A':>12}"
                f"{item['mrr'] if item['mrr'] is not None else 'N/A':>10}"
            )

    print()
    print("JSON report:", json_path)
    print("CSV report:", csv_path)
    print("Database modified: NO")
    print("Paper vectors modified: NO")
    print("Models loaded during summary: NO")


def main_cli():
    parser = argparse.ArgumentParser(
        description="Low-memory, resumable RAG retrieval evaluation."
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=["retrieve", "rerank", "summary"],
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Process only the first N questions for a pilot run.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BACKEND_DIR / "tests" / "retrieval_eval_dataset_v2_en.json"),
        help="Path to the evaluation dataset JSON file.",
    )

    args = parser.parse_args()

    global DATASET_PATH, STAGE1_PATH, STAGE2_PATH

    DATASET_PATH = Path(args.dataset).expanduser().resolve()
    STAGE1_PATH = RESULT_DIR / f"stage1_retrieval_{DATASET_PATH.stem}.json"
    STAGE2_PATH = RESULT_DIR / f"stage2_reranking_{DATASET_PATH.stem}.json"

    print("Dataset:", DATASET_PATH)
    print("Phase 1 checkpoint:", STAGE1_PATH.name)
    print("Phase 2 checkpoint:", STAGE2_PATH.name)

    if args.phase == "retrieve":
        phase_retrieve(args.max_questions)
    elif args.phase == "rerank":
        phase_rerank(args.max_questions)
    else:
        phase_summary()


if __name__ == "__main__":
    main_cli()
