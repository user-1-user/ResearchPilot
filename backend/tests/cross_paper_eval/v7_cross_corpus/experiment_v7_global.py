from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main

V6_PATH = ROOT / "tests/cross_paper_eval/v6/experiment_cross_paper_generalization_v6.py"
OUT_DIR = ROOT / "tests/cross_paper_eval/v7_cross_corpus"
RESULT_DIR = OUT_DIR / "results"

STAGE1_PATH = RESULT_DIR / "v7_stage1_global_retrieval.json"
STAGE2_PATH = RESULT_DIR / "v7_stage2_global_rerank.json"

BASE_WEIGHT = 0.75
RERANK_WEIGHT = 0.25
RRF_K = 60
MAX_PAIR_CHARS = 1800


def load_v6():
    spec = importlib.util.spec_from_file_location("v6_module", V6_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V6 module: {V6_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v6 = load_v6()


def save_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(dataset):
    raw = json.dumps(
        dataset, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_state(state, dataset, stage):
    if state.get("dataset_fingerprint") != fingerprint(dataset):
        raise RuntimeError(f"{stage}: dataset fingerprint mismatch")
    expected = [q["question_id"] for q in dataset["questions"]]
    if state.get("question_ids") != expected:
        raise RuntimeError(f"{stage}: question IDs mismatch")


def load_global_rows():
    rows = []
    papers = {}
    seen = set()

    for pid, info in v6.EXPECTED_PAPERS.items():
        paper, paper_rows, _ = v6.load_paper_rows(pid)
        papers[pid] = paper

        for row in paper_rows:
            cid = row["id"]
            if cid in seen:
                raise RuntimeError(f"Duplicate global chunk ID: {cid}")
            seen.add(cid)
            rows.append(row)

        print(
            f"[LOAD] {info['name']}: {paper['filename']} "
            f"chunks={len(paper_rows)}"
        )

    expected_total = sum(
        info["chunks"] for info in v6.EXPECTED_PAPERS.values()
    )
    if len(rows) != expected_total:
        raise RuntimeError(
            f"Global chunk count mismatch: {len(rows)} != {expected_total}"
        )

    paper_ids = {r["paper_id"] for r in rows}
    if paper_ids != set(v6.EXPECTED_PAPERS):
        raise RuntimeError("Global pool contains unexpected paper IDs")

    print(f"[PASS] Global corpus: {len(rows)} unique chunks")
    return rows, papers


def validate_global_gold(questions, rows):
    ids = {r["id"]: r for r in rows}

    for q in questions:
        for gid in q["gold_chunk_ids"]:
            if gid not in ids:
                raise RuntimeError(
                    f"{q['question_id']}: Gold missing from global corpus: {gid}"
                )
            if ids[gid]["paper_id"] != q["paper_id"]:
                raise RuntimeError(
                    f"{q['question_id']}: Gold belongs to wrong paper: {gid}"
                )

    print("[PASS] All Gold IDs exist in global corpus and have correct paper ownership")


def phase_retrieve(max_questions=None):
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = v6.load_dataset()
    questions = dataset["questions"]
    if max_questions is not None:
        questions = questions[:max_questions]

    rows, papers = load_global_rows()
    validate_global_gold(dataset["questions"], rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if STAGE1_PATH.exists():
        state = load_json(STAGE1_PATH)
        validate_state(state, dataset, "V7 Stage 1")
        results = state.get("results", {})
        print("[RESUME] Stage 1 completed:", len(results))
    else:
        results = {}

    state = {
        "experiment": "V7 global cross-corpus retrieval over all four papers",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": fingerprint(dataset),
        "question_ids": [q["question_id"] for q in dataset["questions"]],
        "corpus": {
            "total_chunks": len(rows),
            "paper_count": len(papers),
            "paper_chunk_counts": {
                v6.EXPECTED_PAPERS[pid]["name"]: sum(
                    1 for r in rows if r["paper_id"] == pid
                )
                for pid in papers
            },
            "scope": "All four papers pooled for every query",
        },
        "retrieval_config": {
            "rrf_k": RRF_K,
            "dense_min_score": v6.DENSE_MIN_SCORE,
            "candidate_rule": "V6 production-faithful BM25 + Dense + RRF eligibility and CJK expansion",
        },
        "results": results,
    }

    print("\nV7 PHASE 1: GLOBAL RETRIEVAL")
    print("Every query searches the same global corpus.")
    print("CrossEncoder is not loaded in this phase.")

    for pos, q in enumerate(questions, 1):
        qid = q["question_id"]
        if qid in results:
            print(f"[{pos}/{len(questions)}] {qid}: already done")
            continue

        output = v6.compute_retrieval_for_question(q, rows)
        candidate_ids = output["candidate_ids"]

        if not candidate_ids:
            raise RuntimeError(f"{qid}: no global eligible candidates")

        id_set = {r["id"] for r in rows}
        if not set(candidate_ids).issubset(id_set):
            raise RuntimeError(f"{qid}: candidate outside global corpus")

        results[qid] = {
            "question_id": qid,
            "target_paper_id": q["paper_id"],
            "target_paper_label": q["paper_label"],
            "gold_chunk_ids": q["gold_chunk_ids"],
            "rrf_candidate_ids": candidate_ids,
            "diagnostic": output["diagnostic"],
        }

        state["results"] = results
        save_json_atomic(STAGE1_PATH, state)

        print(
            f"[{pos}/{len(questions)}] {qid}: "
            f"global_chunks={len(rows)}, "
            f"eligible={output['diagnostic']['eligible_count']}, "
            f"candidates={len(candidate_ids)}, "
            f"target={q['paper_label']}",
            flush=True,
        )
        gc.collect()

    print("\n[PASS] V7 Stage 1 complete")
    print("Saved:", STAGE1_PATH)
    print("Completed:", len(results), "/", len(dataset["questions"]))


def phase_rerank(max_questions=None):
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = v6.load_dataset()
    questions = dataset["questions"]
    if max_questions is not None:
        questions = questions[:max_questions]

    if not STAGE1_PATH.exists():
        raise FileNotFoundError("Run --phase retrieve first")

    stage1 = load_json(STAGE1_PATH)
    validate_state(stage1, dataset, "V7 Stage 1")

    missing = [
        q["question_id"] for q in questions
        if q["question_id"] not in stage1["results"]
    ]
    if missing:
        raise RuntimeError(f"Stage 1 incomplete: {missing[:10]}")

    rows, _ = load_global_rows()
    validate_global_gold(dataset["questions"], rows)
    text_map = {r["id"]: r["text"] for r in rows}
    paper_map = {r["id"]: r["paper_id"] for r in rows}

    if STAGE2_PATH.exists():
        state = load_json(STAGE2_PATH)
        validate_state(state, dataset, "V7 Stage 2")
        results = state.get("results", {})
        print("[RESUME] Stage 2 completed:", len(results))
    else:
        results = {}

    state = {
        "experiment": "V7 global CrossEncoder reranking",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": fingerprint(dataset),
        "question_ids": [q["question_id"] for q in dataset["questions"]],
        "reranker_model": str(main.RERANKER_MODEL),
        "max_pair_chars": MAX_PAIR_CHARS,
        "batch_size": 1,
        "candidate_source": "V7 Stage 1 global RRF candidates",
        "results": results,
    }

    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("Could not load configured CrossEncoder")

    print("\nV7 PHASE 2: GLOBAL CROSS-ENCODER RERANK")
    print("BGE-M3 query model is not loaded in this phase.")
    print("Reranker device:", getattr(reranker, "device", "unknown"))

    for pos, q in enumerate(questions, 1):
        qid = q["question_id"]
        if qid in results:
            print(f"[{pos}/{len(questions)}] {qid}: already done")
            continue

        candidate_ids = stage1["results"][qid]["rrf_candidate_ids"]
        if not candidate_ids:
            raise RuntimeError(f"{qid}: empty candidate list")

        pairs = []
        for cid in candidate_ids:
            if cid not in text_map:
                raise RuntimeError(f"{qid}: unknown candidate {cid}")
            pairs.append([q["question"], text_map[cid][:MAX_PAIR_CHARS]])

        scores = reranker.predict(
            pairs,
            batch_size=1,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )

        if len(scores) != len(candidate_ids):
            raise RuntimeError(f"{qid}: score count mismatch")

        score_pairs = []
        for cid, score in zip(candidate_ids, scores):
            value = float(score)
            if not math.isfinite(value):
                raise RuntimeError(f"{qid}: non-finite reranker score")
            score_pairs.append((cid, value))

        rerank_ids = [
            cid for cid, _ in sorted(
                score_pairs, key=lambda x: x[1], reverse=True
            )
        ]

        results[qid] = {
            "question_id": qid,
            "target_paper_id": q["paper_id"],
            "candidate_ids": candidate_ids,
            "reranker_ranking": rerank_ids,
            "reranker_scores": {
                cid: score for cid, score in score_pairs
            },
            "candidate_paper_ids": {
                cid: paper_map[cid] for cid in candidate_ids
            },
        }

        state["results"] = results
        save_json_atomic(STAGE2_PATH, state)

        print(
            f"[{pos}/{len(questions)}] {qid}: "
            f"reranked={len(candidate_ids)}",
            flush=True,
        )
        del pairs, scores, score_pairs
        gc.collect()

    print("\n[PASS] V7 Stage 2 complete")
    print("Saved:", STAGE2_PATH)
    print("Completed:", len(results), "/", len(dataset["questions"]))


def calculate_global_metrics(ranking, gold_ids):
    return v6.calculate_metrics(ranking, gold_ids)


def paper_error_metrics(ranking, target_pid, paper_by_id):
    output = {}
    for k in (1, 5, 10):
        top = ranking[:k]
        if not top:
            output[f"target_paper_hit@{k}"] = 0.0
            output[f"wrong_paper_fraction@{k}"] = 0.0
            continue

        target_count = sum(
            1 for cid in top if paper_by_id[cid] == target_pid
        )
        wrong_count = len(top) - target_count

        output[f"target_paper_hit@{k}"] = float(target_count > 0)
        output[f"wrong_paper_fraction@{k}"] = wrong_count / len(top)

    output["top1_wrong_paper"] = bool(
        ranking and paper_by_id[ranking[0]] != target_pid
    )
    return output


def phase_summary():
    dataset = v6.load_dataset()

    if not STAGE1_PATH.exists() or not STAGE2_PATH.exists():
        raise FileNotFoundError("Both V7 Stage 1 and Stage 2 are required")

    s1 = load_json(STAGE1_PATH)
    s2 = load_json(STAGE2_PATH)
    validate_state(s1, dataset, "V7 Stage 1")
    validate_state(s2, dataset, "V7 Stage 2")

    questions = dataset["questions"]
    missing1 = [q["question_id"] for q in questions if q["question_id"] not in s1["results"]]
    missing2 = [q["question_id"] for q in questions if q["question_id"] not in s2["results"]]
    if missing1 or missing2:
        raise RuntimeError(
            f"Incomplete stages: Stage1={missing1[:5]}, Stage2={missing2[:5]}"
        )

    rows, _ = load_global_rows()
    paper_by_id = {r["id"]: r["paper_id"] for r in rows}
    all_ids = set(paper_by_id)

    detail = []
    paper_acc = defaultdict(list)

    for q in questions:
        qid = q["question_id"]
        base = s1["results"][qid]["rrf_candidate_ids"]
        rerank = s2["results"][qid]["reranker_ranking"]

        if set(base) != set(rerank) or len(base) != len(rerank):
            raise RuntimeError(f"{qid}: baseline and rerank candidate pools differ")
        if not set(base).issubset(all_ids):
            raise RuntimeError(f"{qid}: candidate outside global corpus")

        hybrid = v6.weighted_rrf(base, rerank)
        gold = q["gold_chunk_ids"]

        bm = calculate_global_metrics(base, gold)
        hm = calculate_global_metrics(hybrid, gold)

        bpaper = paper_error_metrics(base, q["paper_id"], paper_by_id)
        hpaper = paper_error_metrics(hybrid, q["paper_id"], paper_by_id)

        row = {
            "question_id": qid,
            "paper_label": q["paper_label"],
            "target_paper_id": q["paper_id"],
            "global_corpus_size": len(rows),
            "eligible_count": s1["results"][qid]["diagnostic"]["eligible_count"],
            "candidate_count": len(base),
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
            "baseline_target_paper_hit@5": bpaper["target_paper_hit@5"],
            "baseline_target_paper_hit@10": bpaper["target_paper_hit@10"],
            "baseline_wrong_paper_fraction@5": bpaper["wrong_paper_fraction@5"],
            "baseline_wrong_paper_fraction@10": bpaper["wrong_paper_fraction@10"],
            "baseline_top1_wrong_paper": bpaper["top1_wrong_paper"],
            "hybrid_target_paper_hit@5": hpaper["target_paper_hit@5"],
            "hybrid_target_paper_hit@10": hpaper["target_paper_hit@10"],
            "hybrid_wrong_paper_fraction@5": hpaper["wrong_paper_fraction@5"],
            "hybrid_wrong_paper_fraction@10": hpaper["wrong_paper_fraction@10"],
            "hybrid_top1_wrong_paper": hpaper["top1_wrong_paper"],
            "mrr_change": hm["mrr"] - bm["mrr"],
            "recall@5_change": hm["recall@5"] - bm["recall@5"],
            "recall@10_change": hm["recall@10"] - bm["recall@10"],
        }
        detail.append(row)
        paper_acc[q["paper_id"]].append(row)

    summaries = {}
    for pid, info in v6.EXPECTED_PAPERS.items():
        group = paper_acc[pid]

        summaries[info["name"]] = {
            "questions": len(group),
            "baseline": {
                "recall@5": sum(x["baseline_recall@5"] for x in group) / len(group),
                "recall@10": sum(x["baseline_recall@10"] for x in group) / len(group),
                "mrr": sum(x["baseline_mrr"] for x in group) / len(group),
                "top1_wrong_paper_rate": sum(x["baseline_top1_wrong_paper"] for x in group) / len(group),
                "wrong_paper_fraction@5": sum(x["baseline_wrong_paper_fraction@5"] for x in group) / len(group),
                "wrong_paper_fraction@10": sum(x["baseline_wrong_paper_fraction@10"] for x in group) / len(group),
                "target_paper_hit@5": sum(x["baseline_target_paper_hit@5"] for x in group) / len(group),
                "target_paper_hit@10": sum(x["baseline_target_paper_hit@10"] for x in group) / len(group),
            },
            "hybrid": {
                "recall@5": sum(x["hybrid_recall@5"] for x in group) / len(group),
                "recall@10": sum(x["hybrid_recall@10"] for x in group) / len(group),
                "mrr": sum(x["hybrid_mrr"] for x in group) / len(group),
                "top1_wrong_paper_rate": sum(x["hybrid_top1_wrong_paper"] for x in group) / len(group),
                "wrong_paper_fraction@5": sum(x["hybrid_wrong_paper_fraction@5"] for x in group) / len(group),
                "wrong_paper_fraction@10": sum(x["hybrid_wrong_paper_fraction@10"] for x in group) / len(group),
                "target_paper_hit@5": sum(x["hybrid_target_paper_hit@5"] for x in group) / len(group),
                "target_paper_hit@10": sum(x["hybrid_target_paper_hit@10"] for x in group) / len(group),
            },
            "mean_candidate_count": sum(x["candidate_count"] for x in group) / len(group),
        }

    macro = {}
    for method in ("baseline", "hybrid"):
        macro[method] = {}
        for metric in (
            "recall@5", "recall@10", "mrr",
            "top1_wrong_paper_rate",
            "wrong_paper_fraction@5", "wrong_paper_fraction@10",
            "target_paper_hit@5", "target_paper_hit@10",
        ):
            macro[method][metric] = sum(
                summaries[info["name"]][method][metric]
                for info in v6.EXPECTED_PAPERS.values()
            ) / len(v6.EXPECTED_PAPERS)

    improved = sum(x["mrr_change"] > 0 for x in detail)
    worsened = sum(x["mrr_change"] < 0 for x in detail)
    unchanged = len(detail) - improved - worsened

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"v7_global_summary_{timestamp}.json"
    csv_path = RESULT_DIR / f"v7_global_per_question_{timestamp}.csv"

    output = {
        "experiment": "V7 global cross-corpus retrieval and reranking",
        "created_at": timestamp,
        "dataset_name": dataset.get("dataset_name"),
        "dataset_fingerprint": fingerprint(dataset),
        "question_count": len(detail),
        "global_corpus": {
            "total_chunks": len(rows),
            "paper_count": len(v6.EXPECTED_PAPERS),
            "all_queries_use_same_corpus": True,
        },
        "fusion": {
            "formula": "0.75/(60+RRF_rank) + 0.25/(60+CrossEncoder_rank)",
            "baseline_weight": BASE_WEIGHT,
            "reranker_weight": RERANK_WEIGHT,
            "rrf_k": RRF_K,
        },
        "per_paper": summaries,
        "macro_average": macro,
        "mrr_changes": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        },
        "per_question_results": detail,
        "limitations": [
            "Four papers and 80 questions only.",
            "All queries search the same 138-chunk corpus, but production eligibility and candidate caps remain active.",
            "Hybrid reranks only the global RRF candidate set; it cannot recover Gold chunks excluded before reranking.",
            "This experiment evaluates cross-paper interference among these four papers, not generalization to unseen papers.",
            "75/25 fusion weights were fixed from the prior experiment and not tuned on this dataset.",
        ],
    }

    save_json_atomic(json_path, output)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)

    print("\n" + "=" * 100)
    print("V7 GLOBAL CROSS-CORPUS RESULTS")
    print("=" * 100)
    print(f"{'PAPER':<16} {'METHOD':<10} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Wrong@5':>10} {'Wrong@10':>10}")
    for name, summary in summaries.items():
        for method in ("baseline", "hybrid"):
            x = summary[method]
            print(
                f"{name:<16} {method:<10} "
                f"{x['recall@5']:>8.4f} {x['recall@10']:>8.4f} "
                f"{x['mrr']:>8.4f} "
                f"{x['wrong_paper_fraction@5']:>10.4f} "
                f"{x['wrong_paper_fraction@10']:>10.4f}"
            )

    print("-" * 100)
    for method in ("baseline", "hybrid"):
        x = macro[method]
        print(
            f"{'MACRO':<16} {method:<10} "
            f"{x['recall@5']:>8.4f} {x['recall@10']:>8.4f} "
            f"{x['mrr']:>8.4f} "
            f"{x['wrong_paper_fraction@5']:>10.4f} "
            f"{x['wrong_paper_fraction@10']:>10.4f}"
        )

    print("\nMRR changes:", improved, "improved;", worsened, "worsened;", unchanged, "unchanged")
    print("\nSaved:")
    print(json_path.resolve())
    print(csv_path.resolve())


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["retrieve", "rerank", "summary"],
    )
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be >= 1")

    if args.phase == "retrieve":
        phase_retrieve(args.max_questions)
    elif args.phase == "rerank":
        phase_rerank(args.max_questions)
    else:
        phase_summary()


if __name__ == "__main__":
    main_cli()
