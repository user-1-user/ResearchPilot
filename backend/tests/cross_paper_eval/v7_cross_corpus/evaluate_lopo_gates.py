import json
from pathlib import Path

BASE = Path("tests/cross_paper_eval/v7_cross_corpus/results")
SRC = BASE / "v7_prefusion_features_20261117.json"
OUT = BASE / "v7_lopo_gate_results.json"

data = json.loads(SRC.read_text(encoding="utf-8"))
papers = sorted(set(x["paper_id"] for x in data))

rules = {
    "gap>3": lambda x: x["rrf_top1_rank_gap"] > 3,

    "cross_rank>10 AND gap>1": lambda x: (
        x["bm25_top1_rank_in_dense"] > 10
        or x["dense_top1_rank_in_bm25"] > 10
    ) and x["rrf_top1_rank_gap"] > 1,

    "disagree AND gap>0": lambda x: (
        not x["bm25_dense_top1_agree"]
        and x["rrf_top1_rank_gap"] > 0
    ),
}

def metrics(rows, selected_rule=None):
    result = []
    calls = 0

    for x in rows:
        use_ce = selected_rule is not None and selected_rule(x)
        calls += int(use_ce)

        result.append({
            "mrr": x["hybrid_mrr"] if use_ce else x["baseline_mrr"],
            "r5": x["full_hybrid_r5"] if use_ce else x["baseline_r5"],
            "r10": x["full_hybrid_r10"] if use_ce else x["baseline_r10"],
        })

    n = len(result)
    return {
        "n": n,
        "calls": calls,
        "call_rate": calls / n if n else 0,
        "mrr": sum(x["mrr"] for x in result) / n,
        "r5": sum(x["r5"] for x in result) / n,
        "r10": sum(x["r10"] for x in result) / n,
    }

results = []

for held_out in papers:
    train = [x for x in data if x["paper_id"] != held_out]
    test = [x for x in data if x["paper_id"] == held_out]

    candidates = []

    for name, rule in rules.items():
        train_m = metrics(train, rule)
        if train_m["call_rate"] <= 0.40:
            candidates.append((train_m["mrr"], name, rule, train_m))

    # 仅用训练论文选择规则
    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        raise RuntimeError("No candidate satisfies training call-rate limit")

    _, chosen_name, chosen_rule, train_m = candidates[0]

    test_gate = metrics(test, chosen_rule)
    test_base = metrics(test)
    test_full = {
        "mrr": sum(x["hybrid_mrr"] for x in test) / len(test),
        "r5": sum(x["full_hybrid_r5"] for x in test) / len(test),
        "r10": sum(x["full_hybrid_r10"] for x in test) / len(test),
    }

    row = {
        "held_out_paper": held_out,
        "chosen_rule": chosen_name,
        "train": train_m,
        "test_gate": test_gate,
        "test_baseline": test_base,
        "test_full_hybrid": test_full,
    }
    results.append(row)

OUT.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 100)
print("LEAVE-ONE-PAPER-OUT RESULTS")
print("=" * 100)

for x in results:
    g = x["test_gate"]
    b = x["test_baseline"]
    f = x["test_full_hybrid"]

    print("\nHeld out:", x["held_out_paper"])
    print("Selected rule:", x["chosen_rule"])
    print(
        f'Test calls={g["calls"]}/{g["n"]} '
        f'({g["call_rate"]*100:.1f}%)'
    )
    print(
        f'MRR: gate={g["mrr"]:.4f} '
        f'baseline={b["mrr"]:.4f} '
        f'full={f["mrr"]:.4f} '
        f'delta={g["mrr"]-b["mrr"]:+.4f}'
    )
    print(
        f'R5: gate={g["r5"]:.4f} '
        f'baseline={b["r5"]:.4f} '
        f'full={f["r5"]:.4f}'
    )
    print(
        f'R10: gate={g["r10"]:.4f} '
        f'baseline={b["r10"]:.4f} '
        f'full={f["r10"]:.4f}'
    )

print("\nSaved:", OUT)
print("[NOTE] Exploratory LOPO; four-paper sample is still small.")
