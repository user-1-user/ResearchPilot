import sys, json, math, random, importlib.util
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch

ROOT = Path.cwd()
OUT = ROOT / "tests/cross_paper_eval/v8_independent"
DATA = OUT / "v8_independent_eval_candidate.json"
CONFIG = OUT / "v8_gate_config.json"
RESULT = OUT / "v8_gate_evaluation.json"

sys.path.insert(0, str(ROOT))
import main

spec = importlib.util.spec_from_file_location(
    "v6", ROOT / "tests/cross_paper_eval/v6/"
    "experiment_cross_paper_generalization_v6.py"
)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(path)

def rank_map(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {idx: rank for rank, idx in enumerate(order, 1)}

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0

def metric_summary(rows, prefix):
    return {
        "questions": len(rows),
        "mrr": mean(x[prefix]["mrr"] for x in rows),
        "recall@5": mean(x[prefix]["recall@5"] for x in rows),
        "recall@10": mean(x[prefix]["recall@10"] for x in rows),
        "rank1_hits": sum(x[prefix]["rank1_hit"] for x in rows),
    }

def cluster_bootstrap(rows, iterations=5000):
    # 以论文内 Gold Chunk 组合为簇，避免重复问题被当作独立样本
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(sorted(row["gold_chunk_ids"]))
        grouped[row["paper_label"]][key].append(row)

    papers = list(grouped)
    if not papers:
        return {}

    rng = random.Random(20261117)
    output = {}

    for metric in ("mrr", "recall@5", "recall@10"):
        diffs = []
        for _ in range(iterations):
            paper_values = []
            for paper in papers:
                clusters = list(grouped[paper].values())
                sampled = [rng.choice(clusters) for _ in clusters]
                values = []
                for cluster in sampled:
                    values.extend(
                        x["gate"][metric] - x["full"][metric]
                        for x in cluster
                    )
                paper_values.append(mean(values))
            diffs.append(mean(paper_values))

        diffs.sort()
        output[metric] = {
            "mean_gate_minus_full": mean(diffs),
            "ci95": [
                diffs[int(0.025 * (iterations - 1))],
                diffs[int(0.975 * (iterations - 1))]
            ],
            "bootstrap_iterations": iterations,
            "unit": "Gold Chunk cluster, stratified by paper"
        }
    return output

def main_run():
    dataset = load_json(DATA)
    config = load_json(CONFIG)

    if config.get("frozen") is not True:
        raise RuntimeError("Gate config is not frozen")

    gate_cfg = config["gate"]
    if gate_cfg["cross_rank_threshold"] != 10:
        raise RuntimeError("Unexpected cross-rank threshold")
    if gate_cfg["gap_threshold"] != 1:
        raise RuntimeError("Unexpected gap threshold")

    questions = dataset["questions"]
    if len(questions) != 39:
        raise RuntimeError(f"Expected 39 questions, got {len(questions)}")

    if len({q["question_id"] for q in questions}) != len(questions):
        raise RuntimeError("Duplicate question IDs")

    paper_ids = sorted({q["paper_id"] for q in questions})
    rows_by_paper = {}
    text_maps = {}

    print("========== V8 DATA VALIDATION ==========")
    for pid in paper_ids:
        with main.db() as conn:
            paper = conn.execute(
                "SELECT id, user_id, filename FROM papers WHERE id=?",
                (pid,)
            ).fetchone()

        if paper is None:
            raise RuntimeError(f"Paper not found: {pid}")

        rows = main._load_user_chunks(
            paper["user_id"], paper_ids=[pid]
        )
        if not rows:
            raise RuntimeError(f"No chunks found: {pid}")

        ids = {r["id"] for r in rows}
        for q in questions:
            if q["paper_id"] == pid:
                gold = set(q["gold_chunk_ids"])
                if not gold or not gold.issubset(ids):
                    raise RuntimeError(
                        f"{q['question_id']}: invalid Gold Chunk"
                    )

        rows_by_paper[pid] = rows
        text_maps[pid] = {r["id"]: r["text"] for r in rows}
        print(paper["filename"], "chunks =", len(rows))

    print("Questions:", len(questions))
    print("Gate:", gate_cfg["logic"])

    # 第一阶段：生产候选检索 + 预 CrossEncoder Gate 特征
    records = []
    print("\n========== STAGE 1: RETRIEVAL + GATE ==========")

    for pos, q in enumerate(questions, 1):
        pid = q["paper_id"]
        rows = rows_by_paper[pid]
        docs = [r["text"] for r in rows]
        ids = [r["id"] for r in rows]

        # 复用 V6 原始候选生成函数，保留生产候选筛选规则
        retrieval = v6.compute_retrieval_for_question(q, rows)
        base = retrieval["candidate_ids"]
        if not base:
            raise RuntimeError(f"{q['question_id']}: empty candidates")

        encoded = main._embed_texts([q["question"]])
        if len(encoded) != 1:
            raise RuntimeError(f"{q['question_id']}: embedding failed")
        query_vector = [float(x) for x in encoded[0]]

        sparse = main._bm25_scores(q["question"], docs)
        dense = [
            main._cosine(query_vector, json.loads(r["embedding_json"]))
            for r in rows
        ]

        if not any(dense):
            raise RuntimeError(f"{q['question_id']}: dense scores empty")

        sr = rank_map(sparse)
        dr = rank_map(dense)

        bm25_top = max(range(len(rows)), key=lambda i: sparse[i])
        dense_top = max(range(len(rows)), key=lambda i: dense[i])

        # 与 V6 相同的 RRF 排序
        rrf_scores = {}
        for i in range(len(rows)):
            score = 1 / (v6.RRF_K + sr[i])
            score += 1 / (v6.RRF_K + dr[i])
            rrf_scores[i] = score

        rrf_top = max(rrf_scores, key=rrf_scores.get)
        cross_rank = max(
            dr[bm25_top],
            sr[dense_top]
        )
        gap = abs(sr[rrf_top] - dr[rrf_top])

        gate_hit = (
            (dr[bm25_top] > 10 or sr[dense_top] > 10)
            and gap > 1
        )

        records.append({
            "question_id": q["question_id"],
            "paper_id": pid,
            "paper_label": q["paper_label"],
            "gold_chunk_ids": q["gold_chunk_ids"],
            "base_ranking": base,
            "gate_features": {
                "bm25_top1_rank_in_dense": dr[bm25_top],
                "dense_top1_rank_in_bm25": sr[dense_top],
                "rrf_top1_rank_gap": gap,
                "cross_rank_max": cross_rank
            },
            "gate_hit": gate_hit,
            "candidate_count": len(base)
        })

        print(
            f"[{pos}/{len(questions)}] {q['question_id']} "
            f"gate={'HIT' if gate_hit else 'SKIP'} "
            f"candidates={len(base)}",
            flush=True
        )

        del encoded, query_vector, sparse, dense

    # 第二阶段：全量 CrossEncoder 对照
    print("\n========== STAGE 2: FULL CROSS-ENCODER REFERENCE ==========")
    reranker = main._get_reranker()
    if reranker is None:
        raise RuntimeError("CrossEncoder loading failed")

    output_rows = []
    full_predict_calls = 0
    full_pair_count = 0
    gate_predict_calls = 0
    gate_pair_count = 0

    for pos, (q, record) in enumerate(zip(questions, records), 1):
        pid = q["paper_id"]
        candidate_ids = record["base_ranking"]
        text_map = text_maps[pid]

        pairs = []
        for cid in candidate_ids:
            if cid not in text_map:
                raise RuntimeError(f"Unknown candidate: {cid}")
            pairs.append([q["question"], text_map[cid][:1800]])

        scores = reranker.predict(
            pairs,
            batch_size=1,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity()
        )
        if len(scores) != len(candidate_ids):
            raise RuntimeError("CrossEncoder score count mismatch")

        score_pairs = []
        for cid, score in zip(candidate_ids, scores):
            value = float(score)
            if not math.isfinite(value):
                raise RuntimeError("Non-finite CrossEncoder score")
            score_pairs.append((cid, value))

        rerank_ids = [
            cid for cid, _ in sorted(
                score_pairs, key=lambda x: x[1], reverse=True
            )
        ]

        # 全量 CE 对照
        full_ranking = v6.weighted_rrf(candidate_ids, rerank_ids)

        # Gate 路径：命中才运行 CE；未命中直接保留原始检索排序
        if record["gate_hit"]:
            gate_ranking = full_ranking
            gate_predict_calls += 1
            gate_pair_count += len(candidate_ids)
        else:
            gate_ranking = candidate_ids

        gold = q["gold_chunk_ids"]
        full_metrics = v6.calculate_metrics(full_ranking, gold)
        gate_metrics = v6.calculate_metrics(gate_ranking, gold)
        base_metrics = v6.calculate_metrics(candidate_ids, gold)

        output_rows.append({
            "question_id": q["question_id"],
            "paper_id": pid,
            "paper_label": q["paper_label"],
            "gold_chunk_ids": gold,
            "gate_hit": record["gate_hit"],
            "candidate_count": len(candidate_ids),
            "gate_features": record["gate_features"],
            "baseline": base_metrics,
            "full": full_metrics,
            "gate": gate_metrics
        })

        full_predict_calls += 1
        full_pair_count += len(candidate_ids)

        print(
            f"[{pos}/{len(questions)}] {q['question_id']} "
            f"full_MRR={full_metrics['mrr']:.4f} "
            f"gate_MRR={gate_metrics['mrr']:.4f}",
            flush=True
        )

        del pairs, scores, score_pairs
        import gc
        gc.collect()

    # 汇总
    summary = {
        "baseline": metric_summary(output_rows, "baseline"),
        "full_crossencoder": metric_summary(output_rows, "full"),
        "frozen_gate": metric_summary(output_rows, "gate")
    }

    by_paper = {}
    for label in sorted({x["paper_label"] for x in output_rows}):
        subset = [x for x in output_rows if x["paper_label"] == label]
        by_paper[label] = {
            "baseline": metric_summary(subset, "baseline"),
            "full_crossencoder": metric_summary(subset, "full"),
            "frozen_gate": metric_summary(subset, "gate"),
            "gate_selected": sum(x["gate_hit"] for x in subset),
            "question_count": len(subset)
        }

    selected = sum(x["gate_hit"] for x in output_rows)
    total = len(output_rows)

    result = {
        "experiment": "V8 independent paper frozen gate evaluation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATA),
        "dataset_status": dataset.get("status"),
        "manual_review_status": "pending",
        "gate_config": config,
        "question_count": total,
        "paper_count": len(paper_ids),
        "summary": summary,
        "by_paper": by_paper,
        "gate_calls": {
            "selected_questions": selected,
            "skipped_questions": total - selected,
            "selection_rate": selected / total,
            "estimated_call_reduction": 1 - selected / total,
            "gate_path_predict_calls": gate_predict_calls,
            "gate_path_candidate_pairs": gate_pair_count,
            "full_reference_predict_calls": full_predict_calls,
            "full_reference_candidate_pairs": full_pair_count,
            "note": (
                "Full reference pass runs CrossEncoder for all questions. "
                "Gate path counts are counterfactual deployment calls, "
                "not total inference calls consumed by this evaluation."
            )
        },
        "gate_vs_full_cluster_bootstrap": cluster_bootstrap(output_rows),
        "per_question": output_rows
    }

    save_json(RESULT, result)

    print("\n========== V8 FINAL ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nGATE SELECTED:", selected, "/", total)
    print("GATE REDUCTION:", f"{(1-selected/total)*100:.2f}%")
    print("Saved:", RESULT)
    print("[PASS] V8 evaluation completed")

if __name__ == "__main__":
    main_run()
