import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "tests" / "cross_paper_eval" / "results"
DATASET = BASE / "tests" / "cross_paper_eval" / "cross_paper_eval_dataset_v4_reviewed.json"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

dataset = load(DATASET)
stage1 = load(RESULTS / "cross_paper_stage1_retrieval.json")["results"]
stage2 = load(RESULTS / "cross_paper_stage2_rerank.json")["results"]

questions = {q["question_id"]: q for q in dataset["questions"]}

# 从已验证的90题结果中，重新识别纯CE相对RRF的退化题
regressions = []

for qid, q in questions.items():
    base = stage1[qid]["rrf_candidate_ids"]
    ce = stage2[qid]["reranker_ranking"]
    gold = set(q["gold_chunk_ids"])

    base_ranks = {cid: i + 1 for i, cid in enumerate(base)}
    ce_ranks = {cid: i + 1 for i, cid in enumerate(ce)}

    base_rank = min(
        (base_ranks[x] for x in gold if x in base_ranks),
        default=0
    )
    ce_rank = min(
        (ce_ranks[x] for x in gold if x in ce_ranks),
        default=0
    )

    if ce_rank > base_rank:
        regressions.append({
            "qid": qid,
            "base_rank": base_rank,
            "ce_rank": ce_rank,
        })

regressions.sort(key=lambda x: x["qid"])

print("=" * 100)
print("PURE CROSS-ENCODER REGRESSION DIAGNOSTICS")
print("Regression questions:", len(regressions))
print("=" * 100)

for item in regressions:
    qid = item["qid"]
    q = questions[qid]
    s1 = stage1[qid]
    s2 = stage2[qid]

    base = s1["rrf_candidate_ids"]
    ce = s2["reranker_ranking"]
    scores = s2["reranker_scores"]
    gold = set(q["gold_chunk_ids"])

    # 保留候选顺序，生成排名映射
    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    ce_rank = {cid: i + 1 for i, cid in enumerate(ce)}

    print("\n" + "#" * 100)
    print(
        f"{qid} | RRF rank={item['base_rank']} "
        f"| CE rank={item['ce_rank']}"
    )
    print("QUESTION:", q["question"])
    print("PAPER:", q["paper_title"])
    print("GOLD PAGE:", q.get("gold_page_no"))
    print("GOLD IDS:", list(gold))
    print("ANSWER:", q.get("answer", ""))
    print("EVIDENCE QUOTE:", q.get("evidence_quote", ""))

    # Gold 证据在候选池中的位置和分数
    print("\n--- GOLD CHUNKS ---")
    for cid in gold:
        print(
            f"ID={cid} | RRF={base_rank.get(cid, 0)} "
            f"| CE={ce_rank.get(cid, 0)} "
            f"| CE_SCORE={scores.get(cid, 'N/A')}"
        )

    # CE前5名 + Gold（避免重复）
    selected = []
    for cid in ce[:5] + list(gold):
        if cid not in selected:
            selected.append(cid)

    print("\n--- CE TOP 5 + GOLD ---")
    for cid in selected:
        print(
            f"\nCE_RANK={ce_rank.get(cid, 0)} "
            f"RRF_RANK={base_rank.get(cid, 0)} "
            f"SCORE={scores.get(cid, 'N/A')} "
            f"GOLD={'YES' if cid in gold else 'NO'}"
        )
        print("CHUNK_ID:", cid)

        # chunk 原文不在当前缓存中；从数据库读取
        # 后续脚本会使用数据库补全文本
        print("TEXT: [待从 SQLite 数据库读取]")

    print("\n--- RRF TOP 3 ---")
    for cid in base[:3]:
        print(
            f"\nRRF_RANK={base_rank[cid]} "
            f"CE_RANK={ce_rank.get(cid, 0)} "
            f"SCORE={scores.get(cid, 'N/A')} "
            f"GOLD={'YES' if cid in gold else 'NO'}"
        )
        print("CHUNK_ID:", cid)

print("\n" + "=" * 100)
print("CACHE DIAGNOSTICS COMPLETE")
print("Note: chunk text will be added in the next database-backed step.")
