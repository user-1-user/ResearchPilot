import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(".env"))

api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
model = os.getenv("LLM_MODEL", "deepseek-chat")

if not api_key:
    raise RuntimeError(".env中没有配置LLM_API_KEY")

client = OpenAI(api_key=api_key, base_url=base_url)

base = Path("tests/cross_paper_eval/v6/results")
src = base / "language_translation_manual_review_20q.csv"
out = base / "language_translation_llm_review_20q.csv"
json_out = base / "language_translation_llm_review_20q.json"

with src.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 20:
    raise RuntimeError(f"预期20条，实际读取到{len(rows)}条，停止审核")

results = []

for i, r in enumerate(rows, 1):
    prompt = f"""
你是一名严谨的学术论文翻译审校专家。

请判断下面英文问题是否准确表达中文问题。
Gold evidence仅用于理解专业术语，不要求它完整回答问题。
不要因为英文表达方式不同就判错。

中文问题：
{r['zh_question']}

英文翻译：
{r['en_question']}

相关论文证据：
{r.get('gold_text', '')[:5000]}

请严格按照以下标准：
PASS：意思准确，关键术语和限定条件正确。
MINOR：轻微措辞或术语表达问题，不改变核心含义。
FAIL：错译、漏译、关键限定条件缺失或语义改变。
AMBIGUOUS：信息不足，无法可靠判断。

issue_type只能选择：
无、术语、漏译、限定条件、语义偏差、信息不足

只返回JSON，不要输出其他内容：
{{
  "translation_judgment": "PASS/MINOR/FAIL/AMBIGUOUS",
  "issue_type": "上述选项之一",
  "confidence": 0.0,
  "reason": "简短中文理由"
}}
"""

    print(f"[{i}/20] 审核 {r['question_id']} ...", flush=True)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, re.S)

    if not match:
        result = {
            "translation_judgment": "AMBIGUOUS",
            "issue_type": "信息不足",
            "confidence": 0,
            "reason": "模型未返回有效JSON：" + content[:300]
        }
    else:
        try:
            result = json.loads(match.group())
            if result.get("translation_judgment") not in {
                "PASS", "MINOR", "FAIL", "AMBIGUOUS"
            }:
                raise ValueError("无效判断标签")
        except Exception as e:
            result = {
                "translation_judgment": "AMBIGUOUS",
                "issue_type": "信息不足",
                "confidence": 0,
                "reason": f"JSON解析失败：{e}"
            }

    r.update(result)
    r["review_model"] = model
    results.append(r)

    print(
        f"  {result['translation_judgment']} | "
        f"{result.get('issue_type')} | "
        f"{result.get('reason', '')}",
        flush=True
    )

    time.sleep(0.3)

fields = list(rows[0].keys()) + [
    "translation_judgment", "issue_type",
    "confidence", "reason", "review_model"
]

# 去重字段
fields = list(dict.fromkeys(fields))

with out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

with json_out.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n========== 审核统计 ==========")
for label in ["PASS", "MINOR", "FAIL", "AMBIGUOUS"]:
    print(label, sum(
        r["translation_judgment"] == label for r in results
    ))

print("\nCSV:", out.resolve())
print("JSON:", json_out.resolve())
print("审核模型:", model)
