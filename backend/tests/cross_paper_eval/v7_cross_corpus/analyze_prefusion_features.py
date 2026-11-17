import sys
import json
import csv
import gc
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V7_PATH = ROOT / "tests/cross_paper_eval/v7_cross_corpus/experiment_v7_global.py"
spec = importlib.util.spec_from_file_location("v7_gate_module", V7_PATH)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

main = v7.main
v6 = v7.v6

RESULT_DIR = ROOT / "tests/cross_paper_eval/v7_cross_corpus/results"
SELECTIVE_PATH = RESULT_DIR / "v7_selective_b_only_20261117_162736.json"

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def rank_map(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {idx: rank for rank, idx in enumerate(order, 1)}

def margin(scores):
    values = sorted(scores, reverse=True)
    return values[0] - values[1] if len(values) > 1 else 0.0

def average(items, key):
    return sum(float(x[key]) for x in items) / len(items) if items else 0.0

def report(name, items):
    if not items:
        return
    dm = average(items, "hybrid_mrr") - average(items, "baseline_mrr")
    dr5 = average(items, "full_hybrid_r5") - average(items, "baseline_r5")
    dr10 = average(items, "full_hybrid_r10") - average(items, "baseline_r10")
    print(
        f"{name:32} N={len(items):2} "
        f"MRR_delta={dm:+.4f} "
        f"R5_delta={dr5:+.4f} "
        f"R10_delta={dr10:+.4f}"
    )

print("Loading V7 dataset and global corpus...")
dataset = v6.load_dataset()
questions = dataset["questions"]
rows, papers = v7.load_global_rows()
v7.validate_global_gold(questions, rows)

label_data = load_json(SELECTIVE_PATH)
labels = {x["question_id"]: x for x in label_data["questions"]}

if len(labels) != 80:
    raise RuntimeError(f"Expected 80 evaluation labels, got {len(labels)}")

docs = [r["text"] for r in rows]
row_ids = [r["id"] for r in rows]
RRF_K = v7.RRF_K
DENSE_MIN_SCORE = v6.DENSE_MIN_SCORE

output = []

print("\nRecomputing BM25 + Dense only. CrossEncoder is NOT loaded.")

for pos, q in enumerate(questions, 1):
    qid = q["question_id"]
    if qid not in labels:
        raise RuntimeError(f"Missing evaluation label: {qid}")

    encoded = main._embed_texts([q["question"]])
    if len(encoded) != 1:
        raise RuntimeError(f"Embedding failed: {qid}")

    query_vector = [float(x) for x in encoded[0]]
    del encoded

    sparse = main._bm25_scores(q["question"], docs)
    dense = [
        main._cosine(query_vector, json.loads(r["embedding_json"]))
        for r in rows
    ]
    has_dense = bool(query_vector and any(dense))

    srank = rank_map(sparse)
    drank = rank_map(dense)

    fused = []
    for idx in range(len(rows)):
        score = 1 / (RRF_K + srank[idx])
        if has_dense:
            score += 1 / (RRF_K + drank[idx])
        fused.append((score, idx))
    fused.sort(reverse=True)

    rrf_top_idx = fused[0][1]
    sparse_top_idx = max(range(len(rows)), key=lambda i: sparse[i])
    dense_top_idx = max(range(len(rows)), key=lambda i: dense[i])

    # Pre-CrossEncoder features only
    label = labels[qid]
    item = {
        "question_id": qid,
        "paper_id": q["paper_id"],
        "bm25_dense_top1_agree": sparse_top_idx == dense_top_idx,
        "bm25_top1_rank_in_dense": drank[sparse_top_idx],
        "dense_top1_rank_in_bm25": srank[dense_top_idx],
        "rrf_top1_bm25_rank": srank[rrf_top_idx],
        "rrf_top1_dense_rank": drank[rrf_top_idx],
        "rrf_top1_rank_gap": abs(srank[rrf_top_idx] - drank[rrf_top_idx]),
        "bm25_top1_margin": margin(sparse),
        "dense_top1_margin": margin(dense),
        "bm25_top1_score": sparse[sparse_top_idx],
        "dense_top1_score": dense[dense_top_idx],
        "rrf_top1_is_bm25_top1": rrf_top_idx == sparse_top_idx,
        "rrf_top1_is_dense_top1": rrf_top_idx == dense_top_idx,
        "baseline_mrr": label["baseline_mrr"],
        "hybrid_mrr": label["full_hybrid_mrr"],
        "baseline_r5": label["baseline_r5"],
        "full_hybrid_r5": label["full_hybrid_r5"],
        "baseline_r10": label["baseline_r10"],
        "full_hybrid_r10": label["full_hybrid_r10"],
    }
    item["hybrid_mrr_gain"] = item["hybrid_mrr"] - item["baseline_mrr"]
    item["hybrid_r5_gain"] = item["full_hybrid_r5"] - item["baseline_r5"]
    item["hybrid_r10_gain"] = item["full_hybrid_r10"] - item["baseline_r10"]

    output.append(item)

    checkpoint = RESULT_DIR / "v7_prefusion_features_checkpoint.json"
    checkpoint.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{pos:02}/80] {qid} "
        f"agree={item['bm25_dense_top1_agree']} "
        f"RRF ranks=({item['rrf_top1_bm25_rank']},"
        f"{item['rrf_top1_dense_rank']})",
        flush=True,
    )

    del query_vector, sparse, dense, srank, drank, fused
    gc.collect()

print("\n========== PRE-FUSION FEATURE ANALYSIS ==========")

groups = {
    "BM25_Dense_Top1_agree": lambda x: x["bm25_dense_top1_agree"],
    "BM25_Dense_Top1_disagree": lambda x: not x["bm25_dense_top1_agree"],
    "RRF_Top1_BM25_rank<=3": lambda x: x["rrf_top1_bm25_rank"] <= 3,
    "RRF_Top1_BM25_rank>3": lambda x: x["rrf_top1_bm25_rank"] > 3,
    "RRF_Top1_Dense_rank<=3": lambda x: x["rrf_top1_dense_rank"] <= 3,
    "RRF_Top1_Dense_rank>3": lambda x: x["rrf_top1_dense_rank"] > 3,
    "RRF_rank_gap<=3": lambda x: x["rrf_top1_rank_gap"] <= 3,
    "RRF_rank_gap>3": lambda x: x["rrf_top1_rank_gap"] > 3,
    "BM25_top1_cross_rank<=3": lambda x: x["bm25_top1_rank_in_dense"] <= 3,
    "BM25_top1_cross_rank>3": lambda x: x["bm25_top1_rank_in_dense"] > 3,
    "Dense_top1_cross_rank<=3": lambda x: x["dense_top1_rank_in_bm25"] <= 3,
    "Dense_top1_cross_rank>3": lambda x: x["dense_top1_rank_in_bm25"] > 3,
}

for name, fn in groups.items():
    yes = [x for x in output if fn(x)]
    no = [x for x in output if not fn(x)]
    report(name + " YES", yes)
    report(name + " NO", no)

print("\n========== OVERALL ==========")
report("ALL_80", output)

csv_path = RESULT_DIR / "v7_prefusion_features_20261117.csv"
json_path = RESULT_DIR / "v7_prefusion_features_20261117.json"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
    writer.writeheader()
    writer.writerows(output)

json_path.write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n[PASS] Feature extraction completed.")
print("CSV:", csv_path)
print("JSON:", json_path)
print("Questions:", len(output))
print("CrossEncoder loaded: NO")
