import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tests" / "cross_paper_eval"
RESULT_DIR = DATA_DIR / "results"

V4_PATH = DATA_DIR / "cross_paper_eval_dataset_v4_reviewed.json"
V5_PATH = DATA_DIR / "cross_paper_eval_dataset_v5.json"

S1_PATH = RESULT_DIR / "cross_paper_stage1_retrieval.json"
S2_PATH = RESULT_DIR / "cross_paper_stage2_rerank.json"

RRF_K = 60
BOOTSTRAP_N = 10000
SEED = 20261114

PAPERS = {
    "4cd8e5914040": "GPT-3",
    "20c0b7cbf888": "ViT",
    "aa7deee51d3b": "Chemotactic",
}

METHODS = ("RRF", "RRF75_CE25", "RRF50_CE50", "PURE_CE")


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def weighted_fusion(base, rerank, base_weight):
    rerank_weight = 1.0 - base_weight

    base_rank = {cid: i + 1 for i, cid in enumerate(base)}
    rerank_rank = {cid: i + 1 for i, cid in enumerate(rerank)}

    candidates = list(dict.fromkeys(base + rerank))
    scores = {}

    for cid in candidates:
        score = 0.0
        if cid in base_rank:
            score += base_weight / (RRF_K + base_rank[cid])
        if cid in rerank_rank:
            score += rerank_weight / (RRF_K + rerank_rank[cid])
        scores[cid] = score

    return sorted(candidates, key=lambda cid: scores[cid], reverse=True)


def metrics(ranking, gold_ids):
    gold = set(gold_ids)
    if not gold:
        raise ValueError("Empty Gold IDs")

    first_rank = next(
        (i for i, cid in enumerate(ranking, 1) if cid in gold),
        None,
    )

    return {
        "recall@5": len(set(ranking[:5]) & gold) / len(gold),
        "recall@10": len(set(ranking[:10]) & gold) / len(gold),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_gold_rank": first_rank,
        "rank1_hit": bool(ranking and ranking[0] in gold),
    }


def build_rankings(stage1, stage2, qid):
    s1 = stage1["results"][qid]
    s2 = stage2["results"][qid]

    base = s1["rrf_candidate_ids"]
    rerank = s2["reranker_ranking"]

    if len(base) != len(set(base)):
        raise RuntimeError(f"{qid}: duplicate IDs in Stage1")

    if len(rerank) != len(set(rerank)):
        raise RuntimeError(f"{qid}: duplicate IDs in reranker ranking")

    if set(base) != set(rerank):
        raise RuntimeError(f"{qid}: candidate pool mismatch")

    if set(base) != set(s2["candidate_ids"]):
        raise RuntimeError(f"{qid}: Stage2 candidate_ids mismatch")

    return {
        "RRF": base,
        "RRF75_CE25": weighted_fusion(base, rerank, 0.75),
        "RRF50_CE50": weighted_fusion(base, rerank, 0.50),
        "PURE_CE": rerank,
    }


def load_dataset(path):
    data = read_json(path)
    questions = data.get("questions", [])

    if len(questions) != 90:
        raise RuntimeError(f"{path.name}: expected 90 questions, got {len(questions)}")

    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path.name}: duplicate question IDs")

    counts = defaultdict(int)

    for q in questions:
        if q["paper_id"] not in PAPERS:
            raise RuntimeError(f"Unexpected paper: {q['paper_id']}")

        if not q.get("gold_chunk_ids"):
            raise RuntimeError(f"{q['question_id']}: empty Gold")

        counts[q["paper_id"]] += 1

    if any(counts[pid] != 30 for pid in PAPERS):
        raise RuntimeError(f"{path.name}: paper question counts incorrect: {dict(counts)}")

    return data


def summarize(rows):
    output = {}

    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        output[method] = {}

        for pid, paper_name in PAPERS.items():
            subset = [r for r in method_rows if r["paper_id"] == pid]

            output[method][paper_name] = {
                "questions": len(subset),
                "recall@5": mean(r["recall@5"] for r in subset),
                "recall@10": mean(r["recall@10"] for r in subset),
                "mrr": mean(r["mrr"] for r in subset),
                "rank1_count": sum(r["rank1_hit"] for r in subset),
            }

        # 每篇论文等权宏平均
        output[method]["MACRO"] = {
            "questions": len(method_rows),
            "recall@5": mean(
                output[method][name]["recall@5"] for name in PAPERS.values()
            ),
            "recall@10": mean(
                output[method][name]["recall@10"] for name in PAPERS.values()
            ),
            "mrr": mean(
                output[method][name]["mrr"] for name in PAPERS.values()
            ),
            "rank1_count": sum(r["rank1_hit"] for r in method_rows),
        }

    return output


def bootstrap(rows, method):
    by_paper = defaultdict(list)

    for row in rows:
        if row["method"] == method:
            by_paper[row["paper_id"]].append(row)

    rng = random.Random(SEED)
    output = {}

    for metric in ("recall@5", "recall@10", "mrr"):
        diffs = []

        for _ in range(BOOTSTRAP_N):
            paper_diffs = []

            for pid in PAPERS:
                subset = by_paper[pid]
                sample = [rng.choice(subset) for _ in range(len(subset))]
                paper_diffs.append(mean(
                    r[metric] - r["baseline_" + metric] for r in sample
                ))

            diffs.append(mean(paper_diffs))

        diffs.sort()
        point = mean(
            r[metric] - r["baseline_" + metric]
            for r in rows if r["method"] == method
        )

        lo = int((BOOTSTRAP_N - 1) * 0.025)
        hi = int((BOOTSTRAP_N - 1) * 0.975)

        output[metric] = {
            "mean_difference": point,
            "ci_95_low": diffs[lo],
            "ci_95_high": diffs[hi],
            "probability_gt_zero": sum(x > 0 for x in diffs) / BOOTSTRAP_N,
            "iterations": BOOTSTRAP_N,
        }

    return output


def main():
    print("Loading V4/V5 datasets and cached rankings...")

    v4 = load_dataset(V4_PATH)
    v5 = load_dataset(V5_PATH)
    stage1 = read_json(S1_PATH)
    stage2 = read_json(S2_PATH)

    s1 = stage1["results"]
    s2 = stage2["results"]

    v4_questions = {q["question_id"]: q for q in v4["questions"]}
    v5_questions = {q["question_id"]: q for q in v5["questions"]}

    if set(v4_questions) != set(v5_questions):
        raise RuntimeError("V4/V5 question ID sets differ")

    if not set(v5_questions).issubset(s1) or not set(v5_questions).issubset(s2):
        raise RuntimeError("Missing cached question results")

    for qid in v5_questions:
        if v4_questions[qid]["paper_id"] != v5_questions[qid]["paper_id"]:
            raise RuntimeError(f"{qid}: paper ownership differs between V4/V5")

    print("Question IDs:", len(v5_questions))
    print("Stage1 cache:", len(s1))
    print("Stage2 cache:", len(s2))

    all_rows = []
    summaries = {}

    for version, questions in (("V4", v4_questions), ("V5", v5_questions)):
        rows = []

        for qid, q in questions.items():
            rankings = build_rankings(stage1, stage2, qid)
            gold = q["gold_chunk_ids"]

            baseline = metrics(rankings["RRF"], gold)

            for method in METHODS:
                result = metrics(rankings[method], gold)

                row = {
                    "dataset": version,
                    "question_id": qid,
                    "paper_id": q["paper_id"],
                    "paper": PAPERS[q["paper_id"]],
                    "method": method,
                    **result,
                    "baseline_recall@5": baseline["recall@5"],
                    "baseline_recall@10": baseline["recall@10"],
                    "baseline_mrr": baseline["mrr"],
                }

                rows.append(row)
                all_rows.append(row)

        summaries[version] = summarize(rows)

        print(f"\n========== {version} RESULTS ==========")
        for method in METHODS:
            x = summaries[version][method]["MACRO"]
            print(
                f"{method:<14} "
                f"R@5={x['recall@5']:.4f}  "
                f"R@10={x['recall@10']:.4f}  "
                f"MRR={x['mrr']:.4f}  "
                f"Rank1={x['rank1_count']}/90"
            )

        # 逐题比较各方法相对 RRF 的 MRR 变化
        by_q_method = {
            (r["question_id"], r["method"]): r for r in rows
        }

        print("\nCompared with RRF:")
        for method in METHODS:
            if method == "RRF":
                continue

            improved = worsened = unchanged = 0

            for qid in questions:
                a = by_q_method[(qid, method)]["mrr"]
                b = by_q_method[(qid, "RRF")]["mrr"]

                if a > b:
                    improved += 1
                elif a < b:
                    worsened += 1
                else:
                    unchanged += 1

            print(
                f"{method:<14} improved={improved:2d} "
                f"worsened={worsened:2d} unchanged={unchanged:2d}"
            )

    # V5 - V4 指标变化
    print("\n========== V5 MINUS V4 ==========")
    deltas = {}

    for method in METHODS:
        deltas[method] = {}

        for metric in ("recall@5", "recall@10", "mrr"):
            old = summaries["V4"][method]["MACRO"][metric]
            new = summaries["V5"][method]["MACRO"][metric]
            deltas[method][metric] = new - old

        d = deltas[method]
        print(
            f"{method:<14} "
            f"R@5={d['recall@5']:+.4f}  "
            f"R@10={d['recall@10']:+.4f}  "
            f"MRR={d['mrr']:+.4f}"
        )

    # V5 配对分层 Bootstrap：各融合方案相对 RRF
    v5_rows = [r for r in all_rows if r["dataset"] == "V5"]
    significance = {}

    for method in METHODS:
        if method != "RRF":
            significance[method] = bootstrap(v5_rows, method)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "experiment": "Cross-paper V4 vs V5 Gold recomputation",
        "created_at": timestamp,
        "data_source": "Existing Stage1 and Stage2 caches; no model rerun",
        "parameters": {
            "rrf_k": RRF_K,
            "weights": {
                "RRF75_CE25": [0.75, 0.25],
                "RRF50_CE50": [0.50, 0.50],
            },
            "questions": 90,
            "papers": 3,
        },
        "summaries": summaries,
        "v5_minus_v4": deltas,
        "v5_bootstrap_vs_rrf": significance,
    }

    json_path = RESULT_DIR / f"cross_paper_v5_recompute_{timestamp}.json"
    csv_path = RESULT_DIR / f"cross_paper_v5_recompute_{timestamp}.csv"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = list(all_rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n========== BOOTSTRAP: V5 vs RRF ==========")
    for method, values in significance.items():
        print("\n", method)
        for metric, x in values.items():
            print(
                f"{metric:<10} diff={x['mean_difference']:+.4f} "
                f"95%CI=[{x['ci_95_low']:+.4f}, {x['ci_95_high']:+.4f}] "
                f"P(diff>0)={x['probability_gt_zero']:.4f}"
            )

    print("\n========== OUTPUT ==========")
    print("JSON:", json_path)
    print("CSV :", csv_path)
    print("VALIDATION: PASS")


if __name__ == "__main__":
    main()
