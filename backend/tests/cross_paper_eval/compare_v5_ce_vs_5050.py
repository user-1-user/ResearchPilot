import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "tests" / "cross_paper_eval" / "results"

files = sorted(
    RESULT_DIR.glob("cross_paper_v5_recompute_*.csv"),
    key=lambda p: p.stat().st_mtime,
)

if not files:
    raise FileNotFoundError("V5 recompute CSV not found")

csv_path = files[-1]
print("Input:", csv_path.name)

rows = list(csv.DictReader(
    csv_path.open("r", encoding="utf-8-sig", newline="")
))

rows = [r for r in rows if r["dataset"] == "V5"]

data = defaultdict(dict)
paper_names = {}

for r in rows:
    qid = r["question_id"]
    method = r["method"]
    data[qid][method] = r
    paper_names[qid] = r["paper"]

if len(data) != 90:
    raise RuntimeError(f"Expected 90 questions, got {len(data)}")

papers = defaultdict(list)

for qid, methods in data.items():
    if "PURE_CE" not in methods or "RRF50_CE50" not in methods:
        raise RuntimeError(f"{qid}: missing method result")
    papers[paper_names[qid]].append(qid)

if len(papers) != 3 or any(len(qids) != 30 for qids in papers.values()):
    raise RuntimeError("Expected 3 papers with 30 questions each")

metrics = ("recall@5", "recall@10", "mrr")
rng = random.Random(20261115)
N = 10000


def percentile(values, p):
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


result = {
    "comparison": "PURE_CE minus RRF50_CE50",
    "dataset": "V5",
    "questions": 90,
    "papers": dict((k, len(v)) for k, v in papers.items()),
    "iterations": N,
    "metrics": {},
}

print("\n========== PAIRED BOOTSTRAP ==========")

for metric in metrics:
    per_paper = {}

    for paper, qids in papers.items():
        per_paper[paper] = [
            float(data[qid]["PURE_CE"][metric])
            - float(data[qid]["RRF50_CE50"][metric])
            for qid in qids
        ]

    point = mean(mean(values) for values in per_paper.values())

    samples = []
    for _ in range(N):
        paper_diffs = []
        for paper, values in per_paper.items():
            sample = [rng.choice(values) for _ in range(len(values))]
            paper_diffs.append(mean(sample))
        samples.append(mean(paper_diffs))

    samples.sort()

    stats = {
        "mean_difference": point,
        "ci_95_low": percentile(samples, 0.025),
        "ci_95_high": percentile(samples, 0.975),
        "probability_gt_zero": sum(x > 0 for x in samples) / N,
    }

    result["metrics"][metric] = stats

    print(
        f"{metric:<10} diff={point:+.5f} "
        f"95%CI=[{stats['ci_95_low']:+.5f}, "
        f"{stats['ci_95_high']:+.5f}] "
        f"P(CE>50/50)={stats['probability_gt_zero']:.4f}"
    )

# 逐题 MRR 胜负统计
wins = losses = ties = 0

for qid, methods in data.items():
    a = float(methods["PURE_CE"]["mrr"])
    b = float(methods["RRF50_CE50"]["mrr"])

    if a > b:
        wins += 1
    elif a < b:
        losses += 1
    else:
        ties += 1

result["mrr_question_comparison"] = {
    "pure_ce_wins": wins,
    "pure_ce_losses": losses,
    "ties": ties,
}

print("\nMRR question comparison:")
print(f"Pure CE wins={wins}, losses={losses}, ties={ties}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = RESULT_DIR / f"v5_pure_ce_vs_5050_bootstrap_{timestamp}.json"

out.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved:", out)
print("VALIDATION: PASS")
