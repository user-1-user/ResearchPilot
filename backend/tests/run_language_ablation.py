# -*- coding: utf-8 -*-
"""
Three-arm query-language ablation for BGE Reranker.
Run from backend directory:
    .\venv\Scripts\python.exe .\tests\run_language_ablation.py

Read-only inputs:
  tests/retrieval_eval_dataset.json
  tests/retrieval_eval_translations_draft_v1_1.json
  tests/retrieval_eval_results/stage1_retrieval.json
  data/researchpilot.sqlite3

A new timestamped JSON report is created; existing files are never overwritten.
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "retrieval_eval_dataset.json"
TRANSLATIONS_PATH = ROOT / "tests" / "retrieval_eval_translations_draft_v1_1.json"
STAGE1_PATH = ROOT / "tests" / "retrieval_eval_results" / "stage1_retrieval.json"
DB_PATH = ROOT / "data" / "researchpilot.sqlite3"
MODEL_PATH = Path(
    r"C:\Users\Lenovo\.cache\modelscope\models\BAAI--bge-reranker-base\snapshots\master"
)
OUTPUT_DIR = ROOT / "tests" / "retrieval_eval_results"
TEXT_CHAR_LIMIT = 1800
TOP_K_VALUES = (5, 10)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def connect_db_read_only(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到 SQLite 数据库：{path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def macro_recall(rankings_by_qid, gold_by_qid, k):
    values = []
    for qid, gold_ids in gold_by_qid.items():
        if not gold_ids:
            continue
        retrieved = set(rankings_by_qid[qid][:k])
        values.append(len(retrieved.intersection(gold_ids)) / len(gold_ids))
    return sum(values) / len(values) if values else 0.0


def macro_conditional_recall(rankings_by_qid, gold_by_qid, candidates_by_qid, k):
    """Secondary metric: denominator includes only Gold IDs present in Stage1 candidates."""
    values = []
    for qid, gold_ids in gold_by_qid.items():
        present_gold = set(gold_ids).intersection(candidates_by_qid[qid])
        if not present_gold:
            continue
        retrieved = set(rankings_by_qid[qid][:k])
        values.append(len(retrieved.intersection(present_gold)) / len(present_gold))
    return sum(values) / len(values) if values else 0.0


def macro_mrr(rankings_by_qid, gold_by_qid):
    reciprocal_ranks = []
    for qid, gold_ids in gold_by_qid.items():
        rank = next(
            (idx for idx, chunk_id in enumerate(rankings_by_qid[qid], start=1)
             if chunk_id in gold_ids),
            None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def main():
    print("=" * 72)
    print("BGE Reranker 三组查询语言对照实验")
    print("=" * 72)

    for path in (DATASET_PATH, TRANSLATIONS_PATH, STAGE1_PATH, DB_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"必要路径不存在：{path}")

    dataset = load_json(DATASET_PATH)
    translations_data = load_json(TRANSLATIONS_PATH)
    stage1 = load_json(STAGE1_PATH)

    original_questions = {q["id"]: q for q in dataset["questions"]}
    translations = {q["id"]: q for q in translations_data["questions"]}
    qids = [q["id"] for q in dataset["questions"]]

    if len(qids) != 16 or set(qids) != {f"Q{i:02d}" for i in range(1, 17)}:
        raise ValueError("原始数据集不是预期的 Q01-Q16 共 16 题。")
    if set(translations) != set(qids):
        raise ValueError("翻译文件题目 ID 与原始数据集不完全一致。")
    for qid in qids:
        if not translations[qid].get("zh", "").strip() or not translations[qid].get("en", "").strip():
            raise ValueError(f"{qid} 缺少中文或英文查询。")
        if qid not in stage1.get("results", {}):
            raise ValueError(f"Stage1 结果缺少 {qid}。")

    candidates_by_qid = {}
    gold_by_qid = {}
    all_candidate_ids = set()

    for qid in qids:
        stage1_record = stage1["results"][qid]
        candidate_ids = list(dict.fromkeys(stage1_record["rrf_candidate_ids"]))
        if len(candidate_ids) != 28:
            raise ValueError(f"{qid} 的 Stage1 候选数为 {len(candidate_ids)}，预期为 28。")
        candidates_by_qid[qid] = candidate_ids
        gold_by_qid[qid] = set(original_questions[qid]["relevant_chunk_ids"])
        all_candidate_ids.update(candidate_ids)

    print(f"题目数：{len(qids)}")
    print("每题候选数：28")
    print(f"候选 chunk 去重后总数：{len(all_candidate_ids)}")
    print(f"Gold chunk 去重后总数：{len(set().union(*gold_by_qid.values()))}")

    # Fetch all candidate chunks through a read-only SQLite connection.
    conn = connect_db_read_only(DB_PATH)
    try:
        placeholders = ",".join("?" for _ in all_candidate_ids)
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            tuple(all_candidate_ids)
        ).fetchall()
    finally:
        conn.close()

    chunk_texts = {chunk_id: (text or "")[:TEXT_CHAR_LIMIT] for chunk_id, text in rows}
    missing_db_ids = all_candidate_ids - set(chunk_texts)
    if missing_db_ids:
        raise ValueError(f"数据库中找不到 {len(missing_db_ids)} 个候选 chunk，例如：{list(missing_db_ids)[:5]}")

    coverage_by_qid = {}
    for qid in qids:
        present = gold_by_qid[qid].intersection(candidates_by_qid[qid])
        coverage_by_qid[qid] = {
            "gold_total": len(gold_by_qid[qid]),
            "gold_in_stage1_candidates": len(present),
            "gold_missing_from_stage1": sorted(gold_by_qid[qid] - present),
            "stage1_gold_coverage": len(present) / len(gold_by_qid[qid]) if gold_by_qid[qid] else 0.0
        }

    print("\nStage1 Gold 候选覆盖情况：")
    for qid in qids:
        item = coverage_by_qid[qid]
        print(
            f"  {qid}: {item['gold_in_stage1_candidates']}/{item['gold_total']}"
            + (f"；缺失 {len(item['gold_missing_from_stage1'])} 个" if item["gold_missing_from_stage1"] else "")
        )

    print("\n加载 CrossEncoder（CPU，2 线程）...")
    torch.set_num_threads(2)
    model = CrossEncoder(
        str(MODEL_PATH),
        device="cpu",
        activation_fn=torch.nn.Identity()
    )
    effective_max_length = getattr(model, "max_length", None)
    print(f"模型最大输入长度设置：{effective_max_length}")
    print("候选文本统一截取：前 1800 个字符")
    print("每组使用相同的模型、候选集合和评分设置。")
    print("\n开始评分；CPU 环境可能需要一些时间...\n")

    arms = {
        "zh": "中文",
        "en": "英文",
        "bilingual": "中英双语"
    }
    arm_results = {}

    for arm_key, arm_label in arms.items():
        print(f"--- 正在运行：{arm_label} ---")
        rankings_by_qid = {}
        details_by_qid = {}

        for qid in qids:
            zh = translations[qid]["zh"].strip()
            en = translations[qid]["en"].strip()
            query = zh if arm_key == "zh" else en if arm_key == "en" else zh + "\n" + en

            candidate_ids = candidates_by_qid[qid]
            pairs = [[query, chunk_texts[cid]] for cid in candidate_ids]
            scores = model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            scored = [
                {"chunk_id": cid, "score": float(score)}
                for cid, score in zip(candidate_ids, scores)
            ]
            # Python's stable sort preserves Stage1 candidate order in exact score ties.
            scored.sort(key=lambda item: item["score"], reverse=True)
            ranking = [item["chunk_id"] for item in scored]
            rankings_by_qid[qid] = ranking

            gold_ids = gold_by_qid[qid]
            gold_ranks = [
                idx for idx, cid in enumerate(ranking, start=1)
                if cid in gold_ids
            ]
            details_by_qid[qid] = {
                "query": query,
                "gold_ranks": gold_ranks,
                "first_gold_rank": gold_ranks[0] if gold_ranks else None,
                "top10": scored[:10],
                "full_ranking": scored
            }

            rank_text = str(gold_ranks[0]) if gold_ranks else "未命中"
            print(f"  {qid}: 首个 Gold 排名 = {rank_text}")

        metrics = {
            "recall_at_5": macro_recall(rankings_by_qid, gold_by_qid, 5),
            "recall_at_10": macro_recall(rankings_by_qid, gold_by_qid, 10),
            "mrr": macro_mrr(rankings_by_qid, gold_by_qid),
            "conditional_recall_at_5_gold_present_in_stage1": macro_conditional_recall(
                rankings_by_qid, gold_by_qid, candidates_by_qid, 5
            ),
            "conditional_recall_at_10_gold_present_in_stage1": macro_conditional_recall(
                rankings_by_qid, gold_by_qid, candidates_by_qid, 10
            )
        }
        arm_results[arm_key] = {
            "label": arm_label,
            "metrics": metrics,
            "questions": details_by_qid
        }
        print(
            f"  Recall@5={metrics['recall_at_5']:.4f}, "
            f"Recall@10={metrics['recall_at_10']:.4f}, "
            f"MRR={metrics['mrr']:.4f}\n"
        )

    stage1_ceiling = sum(
        item["stage1_gold_coverage"] for item in coverage_by_qid.values()
    ) / len(coverage_by_qid)

    report = {
        "experiment": "Q01-Q16 three-arm query-language ablation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {
            "model": "BAAI/bge-reranker-base",
            "model_path": str(MODEL_PATH),
            "device": "cpu",
            "torch_num_threads": 2,
            "activation_fn": "torch.nn.Identity",
            "candidate_source": "Stage1 RRF rrf_candidate_ids; fixed per question",
            "candidate_count_per_question": 28,
            "chunk_text_char_limit": TEXT_CHAR_LIMIT,
            "batch_size": 16,
            "model_max_length_reported": effective_max_length,
            "query_arms": {
                "zh": "Chinese translation",
                "en": "English translation",
                "bilingual": "Chinese + newline + English"
            }
        },
        "metric_definitions": {
            "recall_at_k": "Macro average across 16 questions: relevant Gold IDs in top-k divided by all annotated Gold IDs for that question. Gold IDs absent from Stage1 remain in the denominator.",
            "mrr": "Macro average reciprocal rank of the first retrieved relevant Gold chunk; zero if no relevant Gold is ranked.",
            "conditional_recall": "Secondary diagnostic metric whose denominator includes only Gold IDs present in that question's Stage1 candidate set."
        },
        "stage1_candidate_coverage": {
            "macro_gold_coverage_ceiling": stage1_ceiling,
            "per_question": coverage_by_qid
        },
        "arms": arm_results,
        "safety": {
            "database_opened_read_only": True,
            "input_files_modified": False,
            "stage1_candidates_modified": False
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"language_ablation_{stamp}.json"
    if output_path.exists():
        raise FileExistsError(f"为避免覆盖，停止写入：{output_path}")
    with output_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("三组实验完成")
    print(f"结果文件：{output_path}")
    print(f"Stage1 平均 Gold 覆盖率上限：{stage1_ceiling:.4f}")
    print("\n最终指标（宏平均）：")
    print(f"{'组别':<12}{'Recall@5':>12}{'Recall@10':>12}{'MRR':>12}")
    for arm_key, arm_label in arms.items():
        m = arm_results[arm_key]["metrics"]
        print(f"{arm_label:<12}{m['recall_at_5']:>12.4f}{m['recall_at_10']:>12.4f}{m['mrr']:>12.4f}")
    print("\n实验只读取原始数据和数据库，并新建结果文件；没有覆盖原始评测文件。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n实验中止：{type(exc).__name__}: {exc}")
        sys.exit(1)
