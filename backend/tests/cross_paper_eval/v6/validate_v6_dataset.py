import json
import sqlite3
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
DATASET = BASE / "v6_eval_dataset_reviewed_80q_corrected.json"
CHUNKS = BASE / "v6_chunks.jsonl"
DB = BASE.parents[2] / "data" / "researchpilot.sqlite3"

expected_papers = {
    "BERT": "a9e1cbc73d75",
    "TorchCurves": "27884946be97",
    "EM-DETR": "92a4b542d44b",
    "Legendre": "a75dab5b1086",
}

data = json.loads(DATASET.read_text(encoding="utf-8"))
chunk_rows = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
chunk_map = {row["id"]: row for row in chunk_rows}

assert len(data["questions"]) == 80, f"Expected 80 questions, got {len(data['questions'])}"
assert len(chunk_map) == 138, f"Expected 138 unique chunks, got {len(chunk_map)}"

ids = [q["question_id"] for q in data["questions"]]
assert len(ids) == len(set(ids)), "Duplicate question_id found"

counts = Counter(q["paper_label"] for q in data["questions"])
assert counts == {"BERT": 20, "TorchCurves": 20, "EM-DETR": 20, "Legendre": 20}, counts

missing = []
wrong_owner = []
empty_gold = []
for q in data["questions"]:
    if not q["gold_chunk_ids"]:
        empty_gold.append(q["question_id"])
    for gid in q["gold_chunk_ids"]:
        row = chunk_map.get(gid)
        if row is None:
            missing.append((q["question_id"], gid))
            continue
        if row.get("paper_label") != q["paper_label"]:
            wrong_owner.append((q["question_id"], gid, row.get("paper_label")))
        if not (row.get("text") or "").strip():
            empty_gold.append((q["question_id"], gid))

assert not missing, f"Missing Gold IDs: {missing}"
assert not wrong_owner, f"Wrong-paper Gold IDs: {wrong_owner}"
assert not empty_gold, f"Empty Gold evidence: {empty_gold}"

conn = sqlite3.connect(DB)
cur = conn.cursor()
db_missing = []
for q in data["questions"]:
    for gid in q["gold_chunk_ids"]:
        found = cur.execute("SELECT paper_id FROM chunks WHERE id=?", (gid,)).fetchone()
        if not found:
            db_missing.append((q["question_id"], gid))
        elif found[0] != expected_papers[q["paper_label"]]:
            wrong_owner.append((q["question_id"], gid, found[0]))
conn.close()

assert not db_missing, f"Gold IDs absent from SQLite: {db_missing}"
assert not wrong_owner, f"SQLite paper ownership mismatch: {wrong_owner}"

print("========== V6 REVIEWED DATASET VALIDATION ==========")
print("Questions:", len(data["questions"]))
print("Paper distribution:", dict(counts))
print("Source chunks:", len(chunk_map))
print("Gold references:", sum(len(q["gold_chunk_ids"]) for q in data["questions"]))
print("Missing source IDs: 0")
print("Wrong paper ownership: 0")
print("SQLite missing Gold IDs: 0")
print("VALIDATION: PASS")
