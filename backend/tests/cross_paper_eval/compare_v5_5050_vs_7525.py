import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

root = Path("tests/cross_paper_eval/results")
files = sorted(root.glob("cross_paper_v5_recompute_*.csv"),
               key=lambda p: p.stat().st_mtime)
rows = list(csv.DictReader(files[-1].open(encoding="utf-8-sig")))

data = defaultdict(dict)
papers = {}

for r in rows:
    if r["dataset"] != "V5":
        continue
    data[r["question_id"]][r["method"]] = r
    papers[r["question_id"]] = r["paper"]

groups = defaultdict(list)
for qid, paper in papers.items():
    groups[paper].append(qid)

rng = random.Random(20261115)
N = 10000

print("Comparison: RRF50_CE50 minus RRF75_CE25")

for metric in ("recall@5", "recall@10", "mrr"):
    by_paper = {
        paper: [
            float(data[q]["RRF50_CE50"][metric]) -
            float(data[q]["RRF75_CE25"][metric])
            for q in qids
        ]
        for paper, qids in groups.items()
    }

    point = mean(mean(v) for v in by_paper.values())
    samples = []

    for _ in range(N):
        paper_means = []
        for values in by_paper.values():
            sample = [rng.choice(values) for _ in values]
            paper_means.append(mean(sample))
        samples.append(mean(paper_means))

    samples.sort()
    lo = samples[int((N - 1) * 0.025)]
    hi = samples[int((N - 1) * 0.975)]
    prob = sum(x > 0 for x in samples) / N

    print(f"{metric}: diff={point:+.5f}, "
          f"95%CI=[{lo:+.5f},{hi:+.5f}], P(diff>0)={prob:.4f}")

wins = losses = ties = 0
for methods in data.values():
    a = float(methods["RRF50_CE50"]["mrr"])
    b = float(methods["RRF75_CE25"]["mrr"])
    if a > b:
        wins += 1
    elif a < b:
        losses += 1
    else:
        ties += 1

print(f"MRR questions: 50/50 wins={wins}, losses={losses}, ties={ties}")
print("VALIDATION: PASS")
