from pathlib import Path
import re

BASE = Path.cwd()
SOURCE = BASE / "tests/cross_paper_eval/experiment_cross_paper_generalization.py"
TARGET = BASE / "tests/cross_paper_eval/v6/experiment_cross_paper_generalization_v6.py"
RESULT_DIR = TARGET.parent / "results_corrected"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not (BASE / "tests/cross_paper_eval/v6/v6_eval_dataset_reviewed_80q_corrected.json").exists():
    raise FileNotFoundError("V6 dataset not found")

s = SOURCE.read_text(encoding="utf-8-sig")

def replace_once(pattern, replacement, label):
    global s
    s2, n = re.subn(pattern, lambda _: replacement, s, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 replacement, got {n}")
    s = s2

# 1. V6 paths, papers and independent result directory
replace_once(
    r'DATA_DIR = BACKEND_DIR.*?STAGE2_PATH = RESULT_DIR / "cross_paper_stage2_rerank.json"',
    '''DATA_DIR = BACKEND_DIR / "tests" / "cross_paper_eval"
V6_DIR = DATA_DIR / "v6"
DATASET_PATH = V6_DIR / "v6_eval_dataset_reviewed_80q_corrected.json"
RESULT_DIR = V6_DIR / "results_corrected"

EXPECTED_PAPERS = {
    "a9e1cbc73d75": {"name": "BERT", "chunks": 58},
    "27884946be97": {"name": "TorchCurves", "chunks": 26},
    "92a4b542d44b": {"name": "EM-DETR", "chunks": 30},
    "a75dab5b1086": {"name": "Legendre", "chunks": 24},
}
VECTOR_DIM = 1024
RRF_K = 60
BASE_WEIGHT = 0.75
RERANK_WEIGHT = 0.25
DENSE_MIN_SCORE = 0.18
BOOTSTRAP_N = 10000
RANDOM_SEED = 20261115

STAGE1_PATH = RESULT_DIR / "cross_paper_stage1_retrieval.json"
STAGE2_PATH = RESULT_DIR / "cross_paper_stage2_rerank.json"''',
    "V6 constants"
)

# 2. Adapt dataset schema
replace_once(
    r'def load_dataset\(\):.*?(?=\ndef load_paper_rows)',
    '''def load_dataset():
    dataset = load_json(DATASET_PATH)
    questions = dataset.get("questions", [])

    if len(questions) != 80:
        raise RuntimeError(f"Expected 80 questions, found {len(questions)}")

    label_to_id = {
        "BERT": "a9e1cbc73d75",
        "TorchCurves": "27884946be97",
        "EM-DETR": "92a4b542d44b",
        "Legendre": "a75dab5b1086",
    }

    qids = [q["question_id"] for q in questions]
    if len(set(qids)) != len(qids):
        raise RuntimeError("Duplicate question_id in dataset")

    counts = defaultdict(int)
    for q in questions:
        for field in ("question_id", "paper_label", "question", "gold_chunk_ids"):
            if field not in q:
                raise RuntimeError(f"{q.get('question_id')}: missing {field}")

        label = q["paper_label"]
        if label not in label_to_id:
            raise RuntimeError(f"{q['question_id']}: unknown paper_label {label}")

        if not q["gold_chunk_ids"]:
            raise RuntimeError(f"{q['question_id']}: empty gold_chunk_ids")

        pid = label_to_id[label]
        q["paper_id"] = pid
        q["paper_title"] = label
        counts[pid] += 1

    if len(counts) != 4 or any(counts[pid] != 20 for pid in EXPECTED_PAPERS):
        raise RuntimeError(f"Expected 20 questions per paper, got {dict(counts)}")

    print("Dataset:", DATASET_PATH.name)
    print("Questions:", len(questions))
    for pid, info in EXPECTED_PAPERS.items():
        print(f"  {info['name']}: {counts[pid]} questions")

    return dataset


''',
    "load_dataset"
)

# 3. Validate Gold ownership; V6 dataset does not contain gold_page_no
replace_once(
    r'def validate_gold\(questions, rows_by_paper\):.*?(?=\ndef )',
    '''def validate_gold(questions, rows_by_paper):
    ids_by_paper = {
        pid: {row["id"]: row for row in rows}
        for pid, rows in rows_by_paper.items()
    }

    for q in questions:
        pid = q["paper_id"]
        for gid in q["gold_chunk_ids"]:
            if gid not in ids_by_paper[pid]:
                raise RuntimeError(
                    f"{q['question_id']}: Gold chunk {gid} missing/wrong paper"
                )


''',
    "validate_gold"
)

# 4. Remove V4-only gold_page_no output field
old = '            "gold_page_no": q["gold_page_no"],\n'
if s.count(old) != 1:
    raise RuntimeError("Could not uniquely locate gold_page_no output field")
s = s.replace(old, "", 1)

# 5. Bootstrap: 20 questions per paper
old = '''    paper_ids = list(EXPECTED_PAPERS)
    if any(len(by_paper[pid]) != 30 for pid in paper_ids):
        raise RuntimeError("Bootstrap requires exactly 30 completed questions per paper")'''
new = '''    paper_ids = list(EXPECTED_PAPERS)
    if any(len(by_paper[pid]) != 20 for pid in paper_ids):
        raise RuntimeError("Bootstrap requires exactly 20 completed questions per paper")'''
if s.count(old) != 1:
    raise RuntimeError("Could not uniquely locate bootstrap 30-question check")
s = s.replace(old, new, 1)

# 6. Report metadata
old = 'info["name"]: {"paper_id": pid, "chunks": info["chunks"], "questions": 30}'
new = 'info["name"]: {"paper_id": pid, "chunks": info["chunks"], "questions": 20}'
if s.count(old) != 1:
    raise RuntimeError("Could not uniquely locate report question count")
s = s.replace(old, new, 1)

s = s.replace(
    "Only three papers and 90 questions were evaluated.",
    "Four papers and 80 questions were evaluated."
)

s = s.replace(
    "this 90-question dataset",
    "this 80-question dataset"
)

# Update experiment description
s = s.replace(
    "Cross-paper generalization: production RRF vs 75/25 weighted RRF-CrossEncoder",
    "V6 cross-paper generalization: production RRF vs 75/25 weighted RRF-CrossEncoder"
)

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(s, encoding="utf-8")

# Syntax validation
compile(s, str(TARGET), "exec")

print("CREATED:", TARGET)
print("SYNTAX: PASS")
print("V4 source file unchanged")
print("V6 result directory:", RESULT_DIR)
