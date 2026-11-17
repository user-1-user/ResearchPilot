import json
import random
from pathlib import Path

p = Path("tests/retrieval_eval_results/selective_rerank_36q_20261114_230046.json")
data = json.loads(p.read_text(encoding="utf-8"))

rows = {}
for r in data["results"]:
    rows.setdefault(r["question_id"], {})[r["method"]] = r

ids = sorted(rows)
rng = random.Random(20261114)
N = 10000

metrics = ["mrr", "recall@5", "recall@10"]

print("Questions:", len(ids))
print("Bootstrap iterations:", N)
print("\nPAIRED BOOTSTRAP: FULL_HYBRID - BASELINE")

for metric in metrics:
    differences = []

    for _ in range(N):
        sample = [rng.choice(ids) for _ in ids]

        base = sum(rows[q]["BASELINE"][metric] for q in sample) / len(sample)
        hybrid = sum(rows[q]["FULL_HYBRID"][metric] for q in sample) / len(sample)

        differences.append(hybrid - base)

    differences.sort()

    mean_diff = sum(differences) / N
    lower = differences[int(N * 0.025)]
    upper = differences[int(N * 0.975)]
    positive = sum(x > 0 for x in differences) / N

    print(f"\n{metric}")
    print(f"Mean difference: {mean_diff:+.5f}")
    print(f"95% bootstrap CI: [{lower:+.5f}, {upper:+.5f}]")
    print(f"P(difference > 0): {positive:.4f}")

print("\nInterpretation: CI excluding zero supports a stable positive difference")
print("within this 36-question evaluation sample.")
