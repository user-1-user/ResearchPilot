import json
from pathlib import Path
from collections import defaultdict

BASE = Path("tests/cross_paper_eval/v7_cross_corpus/results")
SRC = BASE / "v7_prefusion_features_20261117.json"
OUT = BASE / "v7_combined_gate_results.json"

data = json.loads(SRC.read_text(encoding="utf-8"))
N = len(data)

def avg(rows, key):
    return sum(float(x[key]) for x in rows) / len(rows) if rows else 0.0

def evaluate(name, selected):
    selected_ids = {x["question_id"] for x in selected}

    result = []
    for x in data:
        use_ce = x["question_id"] in selected_ids
        result.append({
            "paper_id": x["paper_id"],
            "mrr": x["hybrid_mrr"] if use_ce else x["baseline_mrr"],
            "r5": x["full_hybrid_r5"] if use_ce else x["baseline_r5"],
            "r10": x["full_hybrid_r10"] if use_ce else x["baseline_r10"],
        })

    full = {
        "mrr": avg(data, "hybrid_mrr"),
        "r5": avg(data, "full_hybrid_r5"),
        "r10": avg(data, "full_hybrid_r10"),
    }
    baseline = {
        "mrr": avg(data, "baseline_mrr"),
        "r5": avg(data, "baseline_r5"),
        "r10": avg(data, "baseline_r10"),
    }
    current = {
        "mrr": avg(result, "mrr"),
        "r5": avg(result, "r5"),
        "r10": avg(result, "r10"),
    }

    row = {
        "rule": name,
        "selected": len(selected),
        "call_rate": len(selected) / N,
        "metrics": current,
        "baseline": baseline,
        "full_hybrid": full,
        "delta_vs_baseline": {
            k: current[k] - baseline[k] for k in current
        },
        "loss_vs_full": {
            k: current[k] - full[k] for k in current
        },
        "per_paper": {},
    }

    papers = sorted(set(x["paper_id"] for x in data))
    for pid in papers:
        subset = [x for x in result if x["paper_id"] == pid]
        full_subset = [x for x in data if x["paper_id"] == pid]
        row["per_paper"][pid] = {
            "mrr": avg(subset, "mrr"),
            "full_mrr": avg(full_subset, "hybrid_mrr"),
            "baseline_mrr": avg(full_subset, "baseline_mrr"),
        }

    return row

rules = []

# 规则1：BM25/Dense Top1 不一致 + RRF 排名差距
for gap in [0, 1, 2, 3, 5, 10]:
    selected = [
        x for x in data
        if not x["bm25_dense_top1_agree"]
        and x["rrf_top1_rank_gap"] > gap
    ]
    rules.append(evaluate(f"disagree AND gap>{gap}", selected))

# 规则2：BM25 Top1 在 Dense 排名靠后 + RRF 排名差距
for rank in [3, 5, 10]:
    for gap in [0, 1, 2, 3, 5]:
        selected = [
            x for x in data
            if x["bm25_top1_rank_in_dense"] > rank
            and x["rrf_top1_rank_gap"] > gap
        ]
        rules.append(evaluate(
            f"BM25_top1_dense_rank>{rank} AND gap>{gap}",
            selected
        ))

# 规则3：两个方向任一方跨检索排名靠后 + RRF 排名差距
for rank in [3, 5, 10]:
    for gap in [1, 2, 3, 5]:
        selected = [
            x for x in data
            if (
                x["bm25_top1_rank_in_dense"] > rank
                or x["dense_top1_rank_in_bm25"] > rank
            )
            and x["rrf_top1_rank_gap"] > gap
        ]
        rules.append(evaluate(
            f"cross_rank>{rank} AND gap>{gap}",
            selected
        ))

OUT.write_text(
    json.dumps(rules, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 95)
print("BASELINE:", {k: round(avg(data, v), 4) for k, v in {
    "mrr": "baseline_mrr",
    "r5": "baseline_r5",
    "r10": "baseline_r10"
}.items()})
print("FULL HYBRID:", {k: round(avg(data, v), 4) for k, v in {
    "mrr": "hybrid_mrr",
    "r5": "full_hybrid_r5",
    "r10": "full_hybrid_r10"
}.items()})
print("=" * 95)

for x in sorted(rules, key=lambda z: (z["call_rate"], -z["metrics"]["mrr"])):
    m = x["metrics"]
    d = x["delta_vs_baseline"]
    loss = x["loss_vs_full"]
    print(
        f'{x["rule"]:<48} '
        f'calls={x["selected"]:>2}/{N} ({x["call_rate"]*100:5.1f}%) '
        f'MRR={m["mrr"]:.4f} ({d["mrr"]:+.4f}) '
        f'R5={m["r5"]:.4f} ({d["r5"]:+.4f}) '
        f'R10={m["r10"]:.4f} ({d["r10"]:+.4f}) '
        f'loss_full={loss["mrr"]:+.4f}'
    )

print("\nSaved:", OUT)
print("[NOTE] Retrospective simulation; no actual CrossEncoder calls were saved.")
