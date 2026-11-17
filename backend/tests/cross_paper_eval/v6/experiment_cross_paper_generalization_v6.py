from __future__ import annotations

# Limit CPU threads before importing PyTorch-dependent modules.
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main


DATA_DIR = BACKEND_DIR / "tests" / "cross_paper_eval"
V6_DIR = DATA_DIR / "v6"
DATASET_PATH = V6_DIR / "v6_eval_dataset_reviewed_80q_corrected.json"
RESULT_DIR = V6_DIR / "results_corrected"

EXPECTED_PAPERS = {
    "a9e1cbc73d75": {"name": "BERT", "chunks": 58},
    "27884946be97": {"name": "TorchCurves", "chunks": 26},
    "92a4b542d44b": {"name": "EM-DETR", "chunks": 30},
    "a75dab5b1086": {"name": "Legendre", "chunks": 24},
}
VECTOR_DIM = 1024
RRF_K = 60
BASE_WEIGHT = 0.75
RERANK_WEIGHT = 0.25
DENSE_MIN_SCORE = 0.18
BOOTSTRAP_N = 10000
RANDOM_SEED = 20261115

STAGE1_PATH = RESULT_DIR / "cross_paper_stage1_retrieval.json"
STAGE2_PATH = RESULT_DIR / "cross_paper_stage2_rerank.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def dataset_fingerprint(dataset: dict) -> str:
    raw = json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_dataset():
    dataset = load_json(DATASET_PATH)
    questions = dataset.get("questions", [])

    if len(questions) != 80:
        raise RuntimeError(f"Expected 80 questions, found {len(questions)}")

    label_to_id = {
        "BERT": "a9e1cbc73d75",
        "TorchCurves": "27884946be97",
        "EM-DETR": "92a4b542d44b",
        "Legendre": "a75dab5b1086",
    }

    qids = [q["question_id"] for q in questions]
    if len(set(qids)) != len(qids):
        raise RuntimeError("Duplicate question_id in dataset")

    counts = defaultdict(int)
    for q in questions:
        for field in ("question_id", "paper_label", "question", "gold_chunk_ids"):
            if field not in q:
                raise RuntimeError(f"{q.get('question_id')}: missing {field}")

        label = q["paper_label"]
        if label not in label_to_id:
            raise RuntimeError(f"{q['question_id']}: unknown paper_label {label}")

        if not q["gold_chunk_ids"]:
            raise RuntimeError(f"{q['question_id']}: empty gold_chunk_ids")

        pid = label_to_id[label]
        q["paper_id"] = pid
        q["paper_title"] = label
        counts[pid] += 1

    if len(counts) != 4 or any(counts[pid] != 20 for pid in EXPECTED_PAPERS):
        raise RuntimeError(f"Expected 20 questions per paper, got {dict(counts)}")

    print("Dataset:", DATASET_PATH.name)
    print("Questions:", len(questions))
    for pid, info in EXPECTED_PAPERS.items():
        print(f"  {info['name']}: {counts[pid]} questions")

    return dataset



def load_paper_rows(paper_id: str):
    with main.db() as conn:
        paper = conn.execute(
            "SELECT id, user_id, filename FROM papers WHERE id=?",
            (paper_id,),
        ).fetchone()

    if paper is None:
        raise RuntimeError(f"Paper not found in database: {paper_id}")

    rows = main._load_user_chunks(paper["user_id"], paper_ids=[paper_id])
    expected = EXPECTED_PAPERS[paper_id]["chunks"]
    if len(rows) != expected:
        raise RuntimeError(
            f"{paper_id}: expected {expected} chunks, found {len(rows)}"
        )

    ids = set()
    for row in rows:
        cid = row["id"]
        if cid in ids:
            raise RuntimeError(f"Duplicate chunk id: {cid}")
        ids.add(cid)

        raw = row.get("embedding_json")
        if not raw:
            raise RuntimeError(f"Missing embedding: {cid}")
        vector = json.loads(raw)
        if len(vector) != VECTOR_DIM:
            raise RuntimeError(
                f"{cid}: expected {VECTOR_DIM}-dim vector, got {len(vector)}"
            )
        if not all(math.isfinite(float(x)) for x in vector):
            raise RuntimeError(f"Non-finite embedding value: {cid}")

    return paper, rows, ids


def validate_gold(questions, rows_by_paper):
    ids_by_paper = {
        pid: {row["id"]: row for row in rows}
        for pid, rows in rows_by_paper.items()
    }

    for q in questions:
        pid = q["paper_id"]
        for gid in q["gold_chunk_ids"]:
            if gid not in ids_by_paper[pid]:
                raise RuntimeError(
                    f"{q['question_id']}: Gold chunk {gid} missing/wrong paper"
                )



def validate_state(state, dataset, stage_name):
    if state.get("dataset_fingerprint") != dataset_fingerprint(dataset):
        raise RuntimeError(f"{stage_name}: dataset fingerprint mismatch")
    expected_ids = [q["question_id"] for q in dataset["questions"]]
    if state.get("question_ids") != expected_ids:
        raise RuntimeError(f"{stage_name}: question ID list mismatch")


def get_candidate_count(query: str, rows: list[dict]) -> tuple[int, bool, bool]:
    """Reproduce main.retrieve() production candidate expansion logic."""
    query_tokens = main._retrieval_tokens(query)
    has_cjk_query = any("\u3400" <= char <= "\u9fff" for char in query)
    cjk_query_terms = [
        token for token in query_tokens
        if len(token) >= 2
        and any("\u3400" <= char <= "\u9fff" for char in token)
    ]
    corpus_tokens = {
        token
        for row in rows
        for token in main._retrieval_tokens(row["text"])
    }
    has_cjk_lexical_match = bool(cjk_query_terms) and any(
        token in corpus_tokens for token in cjk_query_terms
    )
    expand = has_cjk_query and not has_cjk_lexical_match
    candidate_cap = 60 if expand else 40
    candidate_count = min(
        max(20, int(main.DEFAULT_TOP_K) * 4, 60 if expand else 20),
        candidate_cap,
    )
    return candidate_count, has_cjk_query, has_cjk_lexical_match


def compute_retrieval_for_question(q, rows):
    """Embedding + BM25 + RRF; uses main._embed_texts() exactly as production."""
    docs = [row["text"] for row in rows]
    query = q["question"]

    encoded = main._embed_texts([query])
    if len(encoded) != 1:
        raise RuntimeError(f"{q['question_id']}: query embedding failed")
    query_vector = [float(x) for x in encoded[0]]
    del encoded

    if len(query_vector) != VECTOR_DIM:
        raise RuntimeError(
            f"{q['question_id']}: query vector dimension {len(query_vector)}"
        )

    sparse = main._bm25_scores(query, docs)
    dense = [
        main._cosine(
            query_vector,
            json.loads(row["embedding_json"]),
        )
        for row in rows
    ]
    has_dense = bool(query_vector and any(dense))

    # Same full-corpus ranks and RRF formula as production.
    sparse_rank = {
        idx: rank
        for rank, idx in enumerate(
            sorted(range(len(rows)), key=lambda i: sparse[i], reverse=True),
            1,
        )
    }
    dense_rank = {
        idx: rank
        for rank, idx in enumerate(
            sorted(range(len(rows)), key=lambda i: dense[i], reverse=True),
            1,
        )
    }

    fused = []
    for idx in range(len(rows)):
        score = 1 / (RRF_K + sparse_rank[idx])
        if has_dense:
            score += 1 / (RRF_K + dense_rank[idx])
        fused.append((score, idx))
    fused.sort(reverse=True)
    fused_score_by_index = {idx: score for score, idx in fused}

    eligible = [
        idx for idx in range(len(rows))
        if sparse[idx] > 0 or (has_dense and dense[idx] >= DENSE_MIN_SCORE)
    ]
    if not eligible:
        return {
            "candidate_ids": [],
            "diagnostic": {
                "eligible_count": 0,
                "candidate_limit": 0,
                "has_dense": has_dense,
                "has_cjk_query": False,
                "has_cjk_lexical_match": False,
            },
        }

    candidate_limit, has_cjk_query, has_cjk_lexical_match = (
        get_candidate_count(query, rows)
    )
    candidate_indices = sorted(
        eligible,
        key=lambda idx: fused_score_by_index[idx],
        reverse=True,
    )[:candidate_limit]

    return {
        "candidate_ids": [rows[idx]["id"] for idx in candidate_indices],
        "diagnostic": {
            "eligible_count": len(eligible),
            "candidate_limit": candidate_limit,
            "actual_candidate_count": len(candidate_indices),
            "has_dense": has_dense,
            "has_cjk_query": has_cjk_query,
            "has_cjk_lexical_match": has_cjk_lexical_match,
        },
    }


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
        questions = questions[:max_questions]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if STAGE1_PATH.exists():
        state = load_json(STAGE1_PATH)
        validate_state(state, dataset, "Stage 1")
        results = state.get("results", {})
        print("Resuming Stage 1 checkpoint:", len(results), "completed")
    else:
        results = {}

    state = {
        "experiment": "Cross-paper production-faithful retrieval stage",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "question_ids": [q["question_id"] for q in dataset["questions"]],
        "production_config": {
            "default_top_k": int(main.DEFAULT_TOP_K),
            "rrf_k": RRF_K,
            "dense_min_score": DENSE_MIN_SCORE,
            "candidate_expansion": "production CJK cross-lingual rule",
        },
        "results": results,
    }

    # Load rows once per paper, and only BGE-M3 in this phase.
    rows_by_paper = {}
    for pid, info in EXPECTED_PAPERS.items():
        paper, rows, _ = load_paper_rows(pid)
        rows_by_paper[pid] = rows
        print(f"{info['name']}: {paper['filename']}, chunks={len(rows)}")

    validate_gold(dataset["questions"], rows_by_paper)

    print("\nPHASE 1: production embedding + BM25 + production RRF")
    print("CrossEncoder is NOT loaded in this phase.")
    print(
        "Embedding backend:",
        "configured API" if main.embedding_client else main.LOCAL_EMBEDDING_MODEL,
    )
    if not main.embedding_client and not main.LOCAL_EMBEDDING_MODEL:
        raise RuntimeError("Neither embedding API nor LOCAL_EMBEDDING_MODEL is configured")
    print()

    for pos, q in enumerate(questions, 1):
        qid = q["question_id"]
        if qid in results:
            print(f"[{pos}/{len(questions)}] {qid}: already done")
            continue

        rows = rows_by_paper[q["paper_id"]]
        output = compute_retrieval_for_question(q, rows)
        candidate_ids = output["candidate_ids"]
        if not candidate_ids:
            raise RuntimeError(
                f"{qid}: production retrieval produced no eligible candidates"
            )

        gold_ids = set(q["gold_chunk_ids"])
        missing_gold = gold_ids - {row["id"] for row in rows}
        if missing_gold:
            raise RuntimeError(f"{qid}: missing Gold IDs {missing_gold}")

        results[qid] = {
            "question_id": qid,
            "paper_id": q["paper_id"],
            "paper_title": q["paper_title"],
            "question": q["question"],
            "gold_chunk_ids": q["gold_chunk_ids"],
            "rrf_candidate_ids": candidate_ids,
            "diagnostic": output["diagnostic"],
        }

        state["results"] = results
        save_json_atomic(STAGE1_PATH, state)

        print(
            f"[{pos}/{len(questions)}] {qid}: "
            f"candidates={len(candidate_ids)}, "
            f"limit={output['diagnostic']['candidate_limit']}, "
            f"CJK={output['diagnostic']['has_cjk_query']}, "
            f"CJK_match={output['diagnostic']['has_cjk_lexical_match']}",
            flush=True,
        )
        gc.collect()

    print("\nPHASE 1 COMPLETE")
    print("Saved:", STAGE1_PATH)
    print("Completed:", len(results), "/", len(dataset["questions"]))
    print("Exit this process before running Phase 2 to release BGE-M3.")


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
        questions = questions[:max_questions]

    if not STAGE1_PATH.exists():
        raise FileNotFoundError("Run --phase retrieve first")
    stage1 = load_json(STAGE1_PATH)
    validate_state(stage1, dataset, "Stage 1")

    stage1_results = stage1.get("results", {})
    missing = [q["question_id"] for q in questions
               if q["question_id"] not in stage1_results]
    if missing:
        raise RuntimeError(f"Stage 1 incomplete; missing: {missing[:10]}")

    rows_by_paper = {}
    text_by_paper = {}
    for pid, info in EXPECTED_PAPERS.items():
        paper, rows, _ = load_paper_rows(pid)
        rows_by_paper[pid] = rows
        text_by_paper[pid] = {row["id"]: row["text"] for row in rows}
        print(f"{info['name']}: loaded {len(rows)} chunks")

    validate_gold(dataset["questions"], rows_by_paper)

    if STAGE2_PATH.exists():
        state = load_json(STAGE2_PATH)
        validate_state(state, dataset, "Stage 2")
        results = state.get("results", {})
        print("Resuming Stage 2 checkpoint:", len(results), "completed")
    else:
        results = {}

    state = {
        "experiment": "Cross-paper CrossEncoder reranking stage",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "question_ids": [q["question_id"] for q in dataset["questions"]],
        "reranker_model": str(main.RERANKER_MODEL),
        "max_pair_chars": 1800,
        "batch_size": 1,
        "results": results,
    }

    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("Could not load configured CrossEncoder reranker")

    print("\nPHASE 2: CrossEncoder only")
    print("BGE-M3 query model is NOT loaded in this phase.")
    print("Reranker device:", getattr(reranker, "device", "unknown"), "\n")

    for pos, q in enumerate(questions, 1):
        qid = q["question_id"]
        if qid in results:
            print(f"[{pos}/{len(questions)}] {qid}: already done")
            continue

        pid = q["paper_id"]
        candidate_ids = stage1_results[qid]["rrf_candidate_ids"]
        text_map = text_by_paper[pid]
        if not candidate_ids:
            raise RuntimeError(f"{qid}: empty candidate list")

        pairs = []
        for cid in candidate_ids:
            if cid not in text_map:
                raise RuntimeError(f"{qid}: candidate not in target paper: {cid}")
            pairs.append([q["question"], text_map[cid][:1800]])

        scores = reranker.predict(
            pairs,
            batch_size=1,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )
        if len(scores) != len(candidate_ids):
            raise RuntimeError(f"{qid}: reranker score count mismatch")

        score_pairs = []
        for cid, score in zip(candidate_ids, scores):
            value = float(score)
            if not math.isfinite(value):
                raise RuntimeError(f"{qid}: non-finite reranker score")
            score_pairs.append((cid, value))

        # Python stable sort preserves candidate order for exact score ties.
        rerank_ids = [
            cid for cid, score in sorted(
                score_pairs, key=lambda item: item[1], reverse=True
            )
        ]

        results[qid] = {
            "question_id": qid,
            "paper_id": pid,
            "candidate_ids": candidate_ids,
            "reranker_ranking": rerank_ids,
            "reranker_scores": {cid: score for cid, score in score_pairs},
        }

        state["results"] = results
        save_json_atomic(STAGE2_PATH, state)
        print(
            f"[{pos}/{len(questions)}] {qid}: reranked {len(candidate_ids)}",
            flush=True,
        )
        del pairs, scores, score_pairs
        gc.collect()

    print("\nPHASE 2 COMPLETE")
    print("Saved:", STAGE2_PATH)
    print("Completed:", len(results), "/", len(dataset["questions"]))
    print("Exit this process before running Phase 3.")


def weighted_rrf(base, rerank):
    """Exact 75/25 weighted reciprocal-rank fusion from prior experiments."""
    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    rerank_rank = {cid: i + 1 for i, cid in enumerate(rerank)}
    candidates = list(dict.fromkeys(base + rerank))
    scores = {}

    for cid in candidates:
        score = 0.0
        if cid in base_rank:
            score += BASE_WEIGHT / (RRF_K + base_rank[cid])
        if cid in rerank_rank:
            score += RERANK_WEIGHT / (RRF_K + rerank_rank[cid])
        scores[cid] = score

    # Stable tie behavior matches the previous selective-rerank script.
    return sorted(candidates, key=lambda cid: scores[cid], reverse=True)


def calculate_metrics(ranking, gold_ids):
    gold = set(gold_ids)
    recall5 = len(set(ranking[:5]) & gold) / len(gold)
    recall10 = len(set(ranking[:10]) & gold) / len(gold)
    first_rank = next(
        (rank for rank, cid in enumerate(ranking, 1) if cid in gold),
        None,
    )
    return {
        "recall@5": recall5,
        "recall@10": recall10,
        "mrr": 1 / first_rank if first_rank else 0.0,
        "first_gold_rank": first_rank,
        "rank1_hit": bool(ranking and ranking[0] in gold),
    }


def mean_metrics(rows):
    if not rows:
        return {}
    return {
        "questions": len(rows),
        "recall@5": mean(r["recall@5"] for r in rows),
        "recall@10": mean(r["recall@10"] for r in rows),
        "mrr": mean(r["mrr"] for r in rows),
        "rank1_count": sum(r["rank1_hit"] for r in rows),
        "top5_count": sum(r["recall@5"] > 0 for r in rows),
        "top10_count": sum(r["recall@10"] > 0 for r in rows),
    }


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * p
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - index) + sorted_values[hi] * (index - lo)


def stratified_paired_bootstrap(per_question, iterations=BOOTSTRAP_N):
    """Resample questions within each paper, then macro-average all 4 papers."""
    rng = random.Random(RANDOM_SEED)
    by_paper = defaultdict(list)
    for row in per_question:
        by_paper[row["paper_id"]].append(row)

    paper_ids = list(EXPECTED_PAPERS)
    if any(len(by_paper[pid]) != 20 for pid in paper_ids):
        raise RuntimeError("Bootstrap requires exactly 20 completed questions per paper")

    output = {}
    for metric in ("recall@5", "recall@10", "mrr"):
        diffs = []
        for _ in range(iterations):
            paper_diffs = []
            for pid in paper_ids:
                rows = by_paper[pid]
                sample = [rng.choice(rows) for _ in range(len(rows))]
                paper_diffs.append(mean(
                    r[f"hybrid_{metric}"] - r[f"baseline_{metric}"]
                    for r in sample
                ))
            diffs.append(mean(paper_diffs))

        diffs.sort()
        output[metric] = {
            "mean_difference": mean(
                r[f"hybrid_{metric}"] - r[f"baseline_{metric}"]
                for r in per_question
            ),
            "ci_95_low": percentile(diffs, 0.025),
            "ci_95_high": percentile(diffs, 0.975),
            "probability_difference_gt_zero": sum(x > 0 for x in diffs) / iterations,
            "iterations": iterations,
            "sampling": "paired, stratified by paper, macro-average across 4 papers",
        }
    return output


def phase_summary():
    dataset = load_dataset()
    if not STAGE1_PATH.exists() or not STAGE2_PATH.exists():
        raise FileNotFoundError("Both Stage 1 and Stage 2 checkpoints are required")

    stage1 = load_json(STAGE1_PATH)
    stage2 = load_json(STAGE2_PATH)
    validate_state(stage1, dataset, "Stage 1")
    validate_state(stage2, dataset, "Stage 2")

    s1 = stage1["results"]
    s2 = stage2["results"]
    questions = dataset["questions"]

    missing1 = [q["question_id"] for q in questions if q["question_id"] not in s1]
    missing2 = [q["question_id"] for q in questions if q["question_id"] not in s2]
    if missing1 or missing2:
        raise RuntimeError(
            f"Incomplete checkpoints: stage1 missing={missing1[:10]}, "
            f"stage2 missing={missing2[:10]}"
        )

    per_question = []
    detail_rows = []

    for q in questions:
        qid = q["question_id"]
        base = s1[qid]["rrf_candidate_ids"]
        rerank = s2[qid]["reranker_ranking"]
        if set(base) != set(rerank):
            raise RuntimeError(f"{qid}: RRF and reranker candidate pools differ")
        if len(base) != len(rerank):
            raise RuntimeError(f"{qid}: candidate count differs")

        hybrid = weighted_rrf(base, rerank)
        bm = calculate_metrics(base, q["gold_chunk_ids"])
        hm = calculate_metrics(hybrid, q["gold_chunk_ids"])
        diag = s1[qid]["diagnostic"]

        row = {
            "question_id": qid,
            "paper_id": q["paper_id"],
            "paper_title": q["paper_title"],
            "candidate_count": len(base),
            "candidate_limit": diag["candidate_limit"],
            "has_cjk_query": diag["has_cjk_query"],
            "has_cjk_lexical_match": diag["has_cjk_lexical_match"],
            "baseline_recall@5": bm["recall@5"],
            "baseline_recall@10": bm["recall@10"],
            "baseline_mrr": bm["mrr"],
            "baseline_first_gold_rank": bm["first_gold_rank"],
            "baseline_rank1_hit": bm["rank1_hit"],
            "hybrid_recall@5": hm["recall@5"],
            "hybrid_recall@10": hm["recall@10"],
            "hybrid_mrr": hm["mrr"],
            "hybrid_first_gold_rank": hm["first_gold_rank"],
            "hybrid_rank1_hit": hm["rank1_hit"],
            "mrr_change": hm["mrr"] - bm["mrr"],
            "recall@5_change": hm["recall@5"] - bm["recall@5"],
            "recall@10_change": hm["recall@10"] - bm["recall@10"],
        }
        per_question.append({
            "paper_id": q["paper_id"],
            "baseline_recall@5": bm["recall@5"],
            "baseline_recall@10": bm["recall@10"],
            "baseline_mrr": bm["mrr"],
            "hybrid_recall@5": hm["recall@5"],
            "hybrid_recall@10": hm["recall@10"],
            "hybrid_mrr": hm["mrr"],
        })
        detail_rows.append(row)

    summaries = {}
    for pid, info in EXPECTED_PAPERS.items():
        paper_rows = [r for r in detail_rows if r["paper_id"] == pid]
        summaries[info["name"]] = {
            "baseline": mean_metrics([{
                "recall@5": r["baseline_recall@5"],
                "recall@10": r["baseline_recall@10"],
                "mrr": r["baseline_mrr"],
                "rank1_hit": r["baseline_rank1_hit"],
            } for r in paper_rows]),
            "hybrid": mean_metrics([{
                "recall@5": r["hybrid_recall@5"],
                "recall@10": r["hybrid_recall@10"],
                "mrr": r["hybrid_mrr"],
                "rank1_hit": r["hybrid_rank1_hit"],
            } for r in paper_rows]),
            "questions": len(paper_rows),
            "candidate_count_mean": mean(r["candidate_count"] for r in paper_rows),
            "candidate_count_min": min(r["candidate_count"] for r in paper_rows),
            "candidate_count_max": max(r["candidate_count"] for r in paper_rows),
        }

    # Macro average: each paper contributes equally, regardless of chunk count.
    macro = {}
    for method in ("baseline", "hybrid"):
        macro[method] = {}
        for metric in ("recall@5", "recall@10", "mrr"):
            macro[method][metric] = mean(
                summaries[info["name"]][method][metric]
                for info in EXPECTED_PAPERS.values()
            )

    bootstrap = stratified_paired_bootstrap(per_question)

    improved = sum(r["mrr_change"] > 0 for r in detail_rows)
    worsened = sum(r["mrr_change"] < 0 for r in detail_rows)
    unchanged = len(detail_rows) - improved - worsened

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"cross_paper_summary_{timestamp}.json"
    csv_path = RESULT_DIR / f"cross_paper_per_question_{timestamp}.csv"

    output = {
        "experiment": "V6 cross-paper generalization: production RRF vs 75/25 weighted RRF-CrossEncoder",
        "created_at": timestamp,
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "question_count": len(detail_rows),
        "papers": {
            info["name"]: {"paper_id": pid, "chunks": info["chunks"], "questions": 20}
            for pid, info in EXPECTED_PAPERS.items()
        },
        "fusion": {
            "formula": "0.75/(60+RRF_rank) + 0.25/(60+CrossEncoder_rank)",
            "baseline_weight": BASE_WEIGHT,
            "reranker_weight": RERANK_WEIGHT,
            "rrf_k": RRF_K,
        },
        "candidate_pool": {
            "rule": "production main.retrieve() eligibility and CJK expansion",
            "dense_min_score": DENSE_MIN_SCORE,
            "per_question_counts": {
                qid: s1[qid]["diagnostic"]["actual_candidate_count"]
                for qid in s1
            },
        },
        "per_paper": summaries,
        "macro_average": macro,
        "paired_bootstrap": bootstrap,
        "mrr_question_changes": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        },
        "per_question_results": detail_rows,
        "limitations": [
            "Four papers and 80 questions were evaluated.",
            "Bootstrap resamples questions within each paper; it does not establish broad paper-level generalization.",
            "The 75/25 fusion weights were fixed from the prior experiment and were not tuned on this 80-question dataset.",
        ],
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_json_atomic(json_path, output)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\n" + "=" * 92)
    print("CROSS-PAPER GENERALIZATION RESULTS")
    print("=" * 92)
    print(f"{'PAPER':<16} {'METHOD':<10} {'R@5':>9} {'R@10':>9} {'MRR':>9} {'Q':>5}")
    for name, s in summaries.items():
        for method in ("baseline", "hybrid"):
            x = s[method]
            print(
                f"{name:<16} {method:<10} {x['recall@5']:>9.4f} "
                f"{x['recall@10']:>9.4f} {x['mrr']:>9.4f} "
                f"{x['questions']:>5}"
            )
    print("-" * 92)
    for method in ("baseline", "hybrid"):
        x = macro[method]
        print(
            f"{'MACRO':<16} {method:<10} {x['recall@5']:>9.4f} "
            f"{x['recall@10']:>9.4f} {x['mrr']:>9.4f} {len(detail_rows):>5}"
        )

    print("\nPaired stratified bootstrap (HYBRID - BASELINE)")
    for metric, result in bootstrap.items():
        print(
            f"{metric}: diff={result['mean_difference']:+.5f}, "
            f"95% CI=[{result['ci_95_low']:+.5f}, "
            f"{result['ci_95_high']:+.5f}], "
            f"P(diff>0)={result['probability_difference_gt_zero']:.4f}"
        )
    print(
        f"\nMRR changes: improved={improved}, worsened={worsened}, "
        f"unchanged={unchanged}"
    )
    print("\nSaved:")
    print(json_path.resolve())
    print(csv_path.resolve())


def main_cli():
    parser = argparse.ArgumentParser(
        description="Cross-paper retrieval and reranking generalization experiment"
    )
    parser.add_argument(
        "--phase",
        choices=("retrieve", "rerank", "summary"),
        required=True,
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional smoke test limit; omit for all 80 questions.",
    )
    args = parser.parse_args()

    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be >= 1")

    if args.phase == "retrieve":
        phase_retrieve(args.max_questions)
    elif args.phase == "rerank":
        phase_rerank(args.max_questions)
    else:
        if args.max_questions is not None:
            parser.error("--max-questions is not supported for summary")
        phase_summary()


if __name__ == "__main__":
    main_cli()
