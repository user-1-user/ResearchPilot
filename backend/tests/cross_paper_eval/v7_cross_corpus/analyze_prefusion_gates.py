import json
from pathlib import Path

root = Path("tests/cross_paper_eval/v7_cross_corpus/results")
s1 = json.loads((root / "v7_stage1_global_retrieval.json").read_text(encoding="utf-8-sig"))
ev = sorted(root.glob("v7_selective_b_only_*.json"))[-1]
rows = json.loads(ev.read_text(encoding="utf-8-sig"))["questions"]

data = []
for r in rows:
    d = s1["results"][r["question_id"]]["diagnostic"]
    data.append({**r, **d})

features = {
    "has_dense": lambda x: x["has_dense"],
    "has_cjk_query": lambda x: x["has_cjk_query"],
    "has_cjk_lexical_match": lambda x: x["has_cjk_lexical_match"],
    "eligible_count>=60": lambda x: x["eligible_count"] >= 60,
    "eligible_count<60": lambda x: x["eligible_count"] < 60,
    "candidate_count=60": lambda x: x["actual_candidate_count"] == 60,
}

def report(name, subset):
    n = len(subset)
    if not n:
        return
    def avg(key):
        return sum(x[key] for x in subset) / n
    print(f"{name:25} n={n:2} "
          f"MRR base={avg('baseline_mrr'):.4f} "
          f"hybrid={avg('full_hybrid_mrr'):.4f} "
          f"delta={avg('full_hybrid_mrr')-avg('baseline_mrr'):+.4f} "
          f"R5 delta={avg('full_hybrid_r5')-avg('baseline_r5'):+.4f}")

print("FEATURE SUBGROUPS")
for name, fn in features.items():
    yes = [x for x in data if fn(x)]
    no = [x for x in data if not fn(x)]
    report(name + " YES", yes)
    report(name + " NO", no)

print("\nALL DATA")
report("ALL", data)
