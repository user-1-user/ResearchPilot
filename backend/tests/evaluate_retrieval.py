from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean


# ------------------------------------------------------------
# 1. Project paths and imports
# ------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main


DATASET_PATH = BACKEND_DIR / "tests" / "retrieval_eval_dataset.json"
RESULT_DIR = BACKEND_DIR / "tests" / "retrieval_eval_results"

PAPER_ID = "11f4d8162178"

# Evaluate Recall@5 and Recall@10.
# top_k=10 mirrors the production retrieve() candidate-pool calculation.
EVAL_TOP_K = 10

# Match the current retrieve() dense eligibility threshold.
DENSE_MIN_SCORE = 0.18

VECTOR_DIM = 1024
EXPECTED_CHUNKS = 38

METHOD_NAMES = [
    "BM25",
    "Dense_BGE_M3",
    "RRF",
    "RRF_plus_BGE_Reranker",
]


# ------------------------------------------------------------
# 2. Load and validate the evaluation dataset
# ------------------------------------------------------------

def load_dataset(dataset_path: Path = DATASET_PATH) -> dict:
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    if dataset.get("paper", {}).get("paper_id") != PAPER_ID:
        raise ValueError("Dataset paper_id does not match target paper.")

    questions = dataset.get("questions", [])

    if not questions:
        raise ValueError("Evaluation dataset contains no questions.")

    question_ids = [q.get("id") for q in questions]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question IDs found.")

    for q in questions:
        if not q.get("question"):
            raise ValueError(f"Empty question: {q.get('id')}")

        if not q.get("relevant_chunk_ids"):
            raise ValueError(
                f"No gold chunk IDs: {q.get('id')}"
            )

    return dataset


# ------------------------------------------------------------
# 3. Load paper chunks and stored BGE-M3 vectors
# ------------------------------------------------------------

def load_paper_data():
    with main.db() as conn:
        paper = conn.execute(
            """
            SELECT id, user_id, filename
            FROM papers
            WHERE id = ?
            """,
            (PAPER_ID,)
        ).fetchone()

    if paper is None:
        raise RuntimeError("Target paper not found in database.")

    rows = main._load_user_chunks(
        paper["user_id"],
        paper_ids=[PAPER_ID],
    )

    if len(rows) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHUNKS} chunks, got {len(rows)}"
        )

    chunk_ids = [row["id"] for row in rows]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise RuntimeError("Duplicate chunk IDs found.")

    vectors = {}

    for row in rows:
        raw = row.get("embedding_json")

        if not raw:
            raise RuntimeError(
                f"Missing embedding: {row['id']}"
            )

        vector = json.loads(raw)

        if len(vector) != VECTOR_DIM:
            raise RuntimeError(
                f"Unexpected vector dimension: {row['id']}"
            )

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in vector
        ):
            raise RuntimeError(
                f"Invalid embedding values: {row['id']}"
            )

        vectors[row["id"]] = vector

    return paper, rows, vectors


# ------------------------------------------------------------
# 4. Calculate retrieval rankings
# ------------------------------------------------------------

def build_rankings(
    query: str,
    query_vector: list[float],
    rows: list[dict],
    vectors: dict,
    reranker,
) -> tuple[dict, dict]:
    docs = [row["text"] for row in rows]

    # BM25 scores.
    sparse_scores = main._bm25_scores(query, docs)

    # Dense cosine scores.
    dense_scores = [
        main._cosine(query_vector, vectors[row["id"]])
        for row in rows
    ]

    has_dense = bool(
        query_vector and any(dense_scores)
    )

    # Independent BM25 ranking.
    bm25_indices = [
        idx
        for idx in range(len(rows))
        if sparse_scores[idx] > 0
    ]

    bm25_indices = sorted(
        bm25_indices,
        key=lambda idx: sparse_scores[idx],
        reverse=True,
    )

    # Independent Dense baseline: rank the complete corpus by cosine score.
    # DENSE_MIN_SCORE is retained only for production-style RRF eligibility.
    dense_indices = (
        sorted(
            range(len(rows)),
            key=lambda idx: dense_scores[idx],
            reverse=True,
        )
        if has_dense
        else []
    )

    # Reciprocal Rank Fusion.
    sparse_rank = {
        idx: rank
        for rank, idx in enumerate(
            sorted(
                range(len(rows)),
                key=lambda idx: sparse_scores[idx],
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
                key=lambda idx: dense_scores[idx],
                reverse=True,
            ),
            1,
        )
    }

    fused = []

    for idx in range(len(rows)):
        score = 1 / (60 + sparse_rank[idx])

        if has_dense:
            score += 1 / (60 + dense_rank[idx])

        fused.append((score, idx))

    # Match production retrieve() ordering.
    fused.sort(reverse=True)

    fused_score_by_index = {
        idx: score
        for score, idx in fused
    }

    # Match production abstention/eligibility conditions.
    eligible = [
        idx
        for idx in range(len(rows))
        if sparse_scores[idx] > 0
        or (
            has_dense
            and dense_scores[idx] >= DENSE_MIN_SCORE
        )
    ]

    candidate_limit = max(
        20,
        min(EVAL_TOP_K * 4, 40)
    )

    rrf_indices = sorted(
        eligible,
        key=lambda idx: fused_score_by_index[idx],
        reverse=True,
    )[:candidate_limit]

    # Rerank the RRF candidate pool.
    # An empty pool is a valid retrieval failure, not a script failure.
    if rrf_indices:
        pairs = [
            [query, rows[idx]["text"][:1800]]
            for idx in rrf_indices
        ]

        import torch

        rerank_scores = reranker.predict(
            pairs,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )

        if len(rerank_scores) != len(rrf_indices):
            raise RuntimeError(
                "CrossEncoder score count mismatch."
            )

        scored_candidates = [
            (float(score), idx)
            for score, idx in zip(rerank_scores, rrf_indices)
        ]

        if not all(
            math.isfinite(score)
            for score, _ in scored_candidates
        ):
            raise RuntimeError(
                "CrossEncoder returned non-finite scores."
            )

        rerank_indices = [
            idx
            for score, idx in sorted(
                scored_candidates,
                key=lambda item: item[0],
                reverse=True,
            )
        ]
    else:
        rerank_indices = []

    rankings = {
        "BM25": [rows[idx]["id"] for idx in bm25_indices],
        "Dense_BGE_M3": [rows[idx]["id"] for idx in dense_indices],
        "RRF": [rows[idx]["id"] for idx in rrf_indices],
        "RRF_plus_BGE_Reranker": [
            rows[idx]["id"] for idx in rerank_indices
        ],
    }

    diagnostic = {
        "bm25_candidates": len(bm25_indices),
        "dense_candidates": len(dense_indices),
        "rrf_candidates": len(rrf_indices),
        "has_dense_query": has_dense,
    }

    return rankings, diagnostic


# ------------------------------------------------------------
# 5. Calculate Recall@k and MRR
# ------------------------------------------------------------

def calculate_metrics(
    ranking: list[str],
    relevant_ids: set[str],
) -> dict:
    top5 = ranking[:5]
    top10 = ranking[:10]

    recall5 = (
        len(set(top5) & relevant_ids) / len(relevant_ids)
    )

    recall10 = (
        len(set(top10) & relevant_ids) / len(relevant_ids)
    )

    first_relevant_rank = None

    for rank, chunk_id in enumerate(ranking, 1):
        if chunk_id in relevant_ids:
            first_relevant_rank = rank
            break

    reciprocal_rank = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    return {
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
    }


# ------------------------------------------------------------
# 6. Macro-average results
# ------------------------------------------------------------

def summarize_results(
    question_results: list[dict],
) -> dict:
    summary = {}

    splits = {
        "all": question_results,
        "zh": [
            row for row in question_results
            if row["language"] == "zh"
        ],
        "en": [
            row for row in question_results
            if row["language"] == "en"
        ],
    }

    for method in METHOD_NAMES:
        summary[method] = {}

        for split_name, split_rows in splits.items():
            values = [
                row["metrics"][method]
                for row in split_rows
            ]

            summary[method][split_name] = {
                "question_count": len(values),
                "recall_at_5": (
                    round(mean(v["recall_at_5"] for v in values), 4)
                    if values else None
                ),
                "recall_at_10": (
                    round(mean(v["recall_at_10"] for v in values), 4)
                    if values else None
                ),
                "mrr": (
                    round(mean(v["reciprocal_rank"] for v in values), 4)
                    if values else None
                ),
            }

    return summary


# ------------------------------------------------------------
# 7. Save JSON and CSV reports
# ------------------------------------------------------------

def save_reports(report: dict, question_results: list[dict]):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    json_path = RESULT_DIR / f"retrieval_eval_{stamp}.json"
    csv_path = RESULT_DIR / f"retrieval_eval_{stamp}.csv"

    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Evaluation result file already exists.")

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_rows = []

    for question in question_results:
        for method in METHOD_NAMES:
            metrics = question["metrics"][method]

            csv_rows.append({
                "question_id": question["id"],
                "language": question["language"],
                "topic": question["topic"],
                "method": method,
                "recall_at_5": round(metrics["recall_at_5"], 6),
                "recall_at_10": round(metrics["recall_at_10"], 6),
                "reciprocal_rank": round(
                    metrics["reciprocal_rank"], 6
                ),
                "first_relevant_rank": (
                    metrics["first_relevant_rank"]
                ),
                "top10_chunk_ids": json.dumps(
                    question["rankings"][method][:10],
                    ensure_ascii=False,
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

    return json_path, csv_path


# ------------------------------------------------------------
# 8. Main evaluation
# ------------------------------------------------------------

def main_eval(dataset_path: Path = DATASET_PATH):
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    dataset = load_dataset(dataset_path)
    questions = dataset["questions"]

    paper, rows, vectors = load_paper_data()

    all_chunk_ids = {row["id"] for row in rows}

    for question in questions:
        missing = (
            set(question["relevant_chunk_ids"]) - all_chunk_ids
        )

        if missing:
            raise RuntimeError(
                f"Gold chunks missing in {question['id']}: {missing}"
            )

    print("Dataset questions:", len(questions))
    print("Paper:", paper["filename"])
    print("Paper chunks:", len(rows))
    print("Stored BGE-M3 vectors:", len(vectors))

    if main.embedding_client:
        embedding_backend = "configured_embedding_client"
    else:
        if not main.LOCAL_EMBEDDING_MODEL:
            raise RuntimeError(
                "No local BGE-M3 model is configured."
            )
        embedding_backend = (
            "local_BGE_M3: " + str(main.LOCAL_EMBEDDING_MODEL)
        )

    print("Embedding backend:", embedding_backend)

    # Encode all questions using the same embedding path as retrieve().
    print()
    print("Encoding evaluation questions...")

    query_vectors = main._embed_texts(
        [question["question"] for question in questions]
    )

    if len(query_vectors) != len(questions):
        raise RuntimeError("Query vector count mismatch.")

    for idx, vector in enumerate(query_vectors):
        if len(vector) != VECTOR_DIM:
            raise RuntimeError(
                f"Query vector dimension mismatch: {questions[idx]['id']}"
            )

        if not all(math.isfinite(float(x)) for x in vector):
            raise RuntimeError(
                f"Invalid query vector: {questions[idx]['id']}"
            )

    print("Query vectors:", len(query_vectors))
    print("Query vector dimension:", VECTOR_DIM)

    # Load the configured CrossEncoder once.
    print()
    print("Loading BGE Reranker...")

    reranker = main._get_reranker()

    if reranker is None:
        raise RuntimeError(
            "BGE Reranker is unavailable. "
            "Cannot complete the four-method comparison."
        )

    print("Reranker loaded:", True)
    print("Reranker device:", getattr(reranker, "device", "unknown"))

    question_results = []

    for position, (question, query_vector) in enumerate(
        zip(questions, query_vectors),
        1,
    ):
        qid = question["id"]

        print(
            f"[{position}/{len(questions)}] Evaluating {qid} "
            f"({question['language']})...",
            flush=True,
        )

        rankings, diagnostic = build_rankings(
            question["question"],
            query_vector,
            rows,
            vectors,
            reranker,
        )

        relevant_ids = set(question["relevant_chunk_ids"])

        method_metrics = {
            method: calculate_metrics(
                rankings[method],
                relevant_ids,
            )
            for method in METHOD_NAMES
        }

        question_results.append({
            "id": qid,
            "language": question["language"],
            "topic": question["topic"],
            "relevant_chunk_count": len(relevant_ids),
            "diagnostic": diagnostic,
            "rankings": rankings,
            "metrics": method_metrics,
        })

    summary = summarize_results(question_results)

    report = {
        "report_name": "RAG Retrieval Evaluation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "query_language": dataset.get("query_language"),
        "query_translation_status": dataset.get("query_translation_status"),
        "source_dataset": dataset.get("source_dataset"),
        "source_translation_file": dataset.get("source_translation_file"),
        "annotation_status": dataset["annotation_policy"]["status"],
        "paper_id": PAPER_ID,
        "paper_filename": paper["filename"],
        "document_count": len(rows),
        "question_count": len(questions),
        "evaluation_top_k": EVAL_TOP_K,
        "dense_min_score": DENSE_MIN_SCORE,
        "rrf_constant": 60,
        "embedding_backend": embedding_backend,
        "reranker_model": str(main.RERANKER_MODEL),
        "methods": METHOD_NAMES,
        "metric_definition": {
            "recall_at_k": (
                "Relevant chunks retrieved in top-k divided "
                "by all gold relevant chunks for that question."
            ),
            "mrr": (
                "Mean reciprocal rank of the first relevant chunk."
            ),
            "averaging": (
                "Macro-average across questions; also grouped by language."
            ),
        },
        "summary": summary,
        "per_question_results": question_results,
    }

    json_path, csv_path = save_reports(
        report,
        question_results,
    )

    # Print aggregate results.
    print()
    print("=" * 90)
    print("AGGREGATE RESULTS (MACRO AVERAGE)")
    print("=" * 90)

    for split_name in ("all", "zh", "en"):
        print()
        print("GROUP:", split_name.upper())

        print(
            f"{'Method':<27}"
            f"{'Questions':>10}"
            f"{'Recall@5':>12}"
            f"{'Recall@10':>12}"
            f"{'MRR':>10}"
        )

        print("-" * 75)

        for method in METHOD_NAMES:
            values = summary[method][split_name]

            recall5 = (
                f"{values['recall_at_5']:.4f}"
                if values["recall_at_5"] is not None else "N/A"
            )
            recall10 = (
                f"{values['recall_at_10']:.4f}"
                if values["recall_at_10"] is not None else "N/A"
            )
            mrr = (
                f"{values['mrr']:.4f}"
                if values["mrr"] is not None else "N/A"
            )
            print(
                f"{method:<27}"
                f"{values['question_count']:>10}"
                f"{recall5:>12}"
                f"{recall10:>12}"
                f"{mrr:>10}"
            )

    print()
    print("=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)
    print("JSON report:", json_path.resolve())
    print("CSV report:", csv_path.resolve())
    print("Database modified: NO")
    print("Paper vectors modified: NO")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResearchPilot RAG retrieval."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the evaluation dataset JSON.",
    )
    args = parser.parse_args()
    main_eval(args.dataset)
