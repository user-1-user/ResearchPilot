import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "tests/cross_paper_eval/v7_cross_corpus/results"
data = json.loads((P / "v7_prefusion_features_20261117.json").read_text(encoding="utf-8"))

rules = {
    "gap>3": lambda x: x["rrf_top1_rank_gap"] > 3,
    "gap>5": lambda x: x["rrf_top1_rank_gap"] > 5,
    "gap>10": lambda x: x["rrf_top1_rank_gap"] > 10,
}

def avg(rows, key):
    return sum(x[key] for x in rows) / len(rows) if rows else 0

def metrics(rows):
    return {
        "mrr": avg(rows, "baseline_mrr"),
        "r5": avg(rows, "baseline_r5"),
        "r10": avg(rows, "baseline_r10"),
    }

def evaluate(name, selected):
    # 被选中问题采用全量Hybrid结果；未选中问题保持RRF结果
    result = []
    for x in data:
        use = x in selected
        result.append({
            "question_id": x["question_id"],
            "paper_id": x["paper_id"],
            "mrr": x["hybrid_mrr"] if use else x["baseline_mrr"],
            "r5": x["full_hybrid_r5"] if use else x["baseline_r5"],
            "r10": x["full_hybrid_r10"] if use else x["baseline_r10"],
        })

    full = [{
        "paper_id": x["paper_id"],
        "mrr": x["hybrid_mrr"],
        "r5": x["full_hybrid_r5"],
        "r10": x["full_hybrid_r10"],
    } for x in data]

    n = len(selected)
    print(f"\n{name}: selected={n}/80 ({n/80:.1%}), skipped={80-n}")
    for key in ("mrr", "r5", "r10"):
        value = avg(result, key)
        full_value = avg(full, key)
        print(f"  {key}: {value:.4f} | full hybrid={full_value:.4f} | loss={value-full_value:+.4f}")

    print("  Per-paper MRR:")
    for pid in sorted(set(x["paper_id"] for x in data)):
        subset = [x for x in result if x["paper_id"] == pid]
        full_subset = [x for x in full if x["paper_id"] == pid]
        print(f"    {pid}: {avg(subset,'mrr'):.4f} | full={avg(full_subset,'mrr'):.4f}")

    return result

all_selected = []
for name, fn in rules.items():
    selected = [x for x in data if fn(x)]
    all_selected.extend(evaluate(name, selected))

print("\n[NOTE] Retrospective simulation only; no CrossEncoder calls were actually saved.")
