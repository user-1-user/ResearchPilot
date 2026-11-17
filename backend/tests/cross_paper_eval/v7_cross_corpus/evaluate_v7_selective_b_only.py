import json, csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path("tests/cross_paper_eval/v7_cross_corpus/results")
S1 = json.loads((ROOT / "v7_stage1_global_retrieval.json").read_text(encoding="utf-8-sig"))
S2 = json.loads((ROOT / "v7_stage2_global_rerank.json").read_text(encoding="utf-8-sig"))

def metrics(ranking, gold):
    gold = set(gold)
    rank = next((i for i, x in enumerate(ranking, 1) if x in gold), None)
    return {
        "mrr": 1 / rank if rank else 0.0,
        "r5": int(any(x in gold for x in ranking[:5])),
        "r10": int(any(x in gold for x in ranking[:10])),
        "rank1": int(bool(ranking) and ranking[0] in gold),
    }

def weighted_rrf(base, rerank):
    br = {x: i for i, x in enumerate(base, 1)}
    rr = {x: i for i, x in enumerate(rerank, 1)}
    return sorted(base, key=lambda x: (
        -(0.75 / (60 + br[x]) + 0.25 / (60 + rr[x])), br[x]
    ))

rows = []
for qid in S1["question_ids"]:
    a, b = S1["results"][qid], S2["results"][qid]
    base, rerank, gold = a["rrf_candidate_ids"], b["reranker_ranking"], a["gold_chunk_ids"]

    if set(base) != set(rerank) or len(base) != len(rerank):
        raise RuntimeError(f"{qid}: candidate pool mismatch")

    # B-only: RRF Top1 is ranked below 3 by CrossEncoder.
    rerank_rank = rerank.index(base[0]) + 1
    selected = rerank_rank > 3

    hybrid = weighted_rrf(base, rerank)
    selective = hybrid if selected else base

    row = {
        "question_id": qid,
        "paper": a.get("target_paper_label", qid.split("_")[0]),
        "rrf_top1_reranker_rank": rerank_rank,
        "b_only_selected": selected,
    }
    for name, ranking in [
        ("baseline", base), ("full_hybrid", hybrid), ("selective_b", selective)
    ]:
        for k, v in metrics(ranking, gold).items():
            row[f"{name}_{k}"] = v
    rows.append(row)

def summarize(items, prefix):
    return {
        key: round(sum(x[f"{prefix}_{key}"] for x in items) / len(items), 6)
        for key in ["mrr", "r5", "r10", "rank1"]
    }

selected = sum(x["b_only_selected"] for x in rows)
summary = {
    "experiment": "V7 B-only fusion selection; retrospective evaluation",
    "question_count": len(rows),
    "selected_questions": selected,
    "selected_ratio": round(selected / len(rows), 4),
    "metrics": {
        name: summarize(rows, name)
        for name in ["baseline", "full_hybrid", "selective_b"]
    },
    "delta_vs_baseline": {},
    "delta_vs_full_hybrid": {},
    "per_paper": {},
    "important_caveat": "B-only requires CrossEncoder ranking, so it does not reduce CrossEncoder inference calls. It only gates fusion after reranking."
}
for key in ["mrr", "r5", "r10", "rank1"]:
    summary["delta_vs_baseline"][key] = round(
        summary["metrics"]["selective_b"][key] - summary["metrics"]["baseline"][key], 6)
    summary["delta_vs_full_hybrid"][key] = round(
        summary["metrics"]["selective_b"][key] - summary["metrics"]["full_hybrid"][key], 6)

groups = defaultdict(list)
for row in rows:
    groups[row["paper"]].append(row)
for paper, items in sorted(groups.items()):
    summary["per_paper"][paper] = {
        name: summarize(items, name)
        for name in ["baseline", "full_hybrid", "selective_b"]
    }

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = ROOT / f"v7_selective_b_only_{stamp}.json"
csv_path = ROOT / f"v7_selective_b_only_{stamp}.csv"
json_path.write_text(json.dumps({"summary": summary, "questions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSAVED:", json_path)
print("SAVED:", csv_path)
