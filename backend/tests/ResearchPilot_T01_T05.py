# ResearchPilot 省赛核心验收：T01-T05
# 放到 backend/tests/ 或任意位置运行；只会在独立测试账号下上传文件，不删除现有论文。
# 用法见脚本下方 PowerShell 命令。

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


def parse_args():
    p = argparse.ArgumentParser(description="ResearchPilot T01-T05 competition acceptance tests")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--v7", type=Path, help="官方 Attention Is All You Need v7 PDF")
    p.add_argument("--v1", type=Path, help="官方 Attention Is All You Need v1 PDF")
    p.add_argument("--scan", type=Path, help="官方扫描版 1706.03762v1_img.pdf")
    p.add_argument("--book", type=Path, help="官方 541 页 Understanding Deep Learning PDF")
    return p.parse_args()


def show(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))
    return ok


def main():
    args = parse_args()
    base = args.base_url.rstrip("/")
    results: list[bool] = []
    paper_ids: list[str] = []
    v7_id = None

    client = httpx.Client(timeout=httpx.Timeout(900.0, connect=15.0))

    def upload(path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        with path.open("rb") as f:
            r = client.post(
                f"{base}/api/papers/upload",
                headers=auth,
                files={"file": (path.name, f, "application/pdf")},
            )
        if r.status_code != 200:
            raise RuntimeError(f"上传失败 HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("paper_id"):
            paper_ids.append(data["paper_id"])
        return data

    try:
        # 服务与隔离测试账号
        h = client.get(f"{base}/api/health")
        h.raise_for_status()
        health = h.json()
        show("服务健康检查", health.get("status") == "ok", json.dumps(health, ensure_ascii=False))
        results.append(health.get("status") == "ok")

        stamp = time.strftime("%Y%m%d%H%M%S")
        username = f"rp_test_{stamp}"
        password = f"RPtest-{stamp}-Aa9!"
        reg = client.post(f"{base}/api/auth/register", json={"username": username, "password": password})
        if reg.status_code != 200:
            raise RuntimeError(f"测试账号注册失败：{reg.status_code} {reg.text[:300]}")
        login = client.post(f"{base}/api/auth/login", json={"username": username, "password": password})
        login.raise_for_status()
        auth = {"Authorization": f"Bearer {login.json()['token']}"}
        print(f"测试账号：{username}（本轮独立账号，不影响原账号论文库）")

        # T01：arXiv v7 -> v1 版本去重与版本差异
        if args.v7 and args.v1:
            try:
                d7 = upload(args.v7)
                v7_id = d7["paper_id"]
                d1 = upload(args.v1)
                duplicates = d1.get("duplicates") or []
                matched = any(x.get("paper_id") == v7_id for x in duplicates)
                v7_ver = (d7.get("metadata") or {}).get("arxiv_version")
                v1_ver = (d1.get("metadata") or {}).get("arxiv_version")
                newer_hint = any("现有版本更新" in str(x.get("suggestion", "")) for x in duplicates if x.get("paper_id") == v7_id)
                ok = bool(matched and v7_ver == 7 and v1_ver == 1 and newer_hint)
                detail = f"v7={v7_ver}, v1={v1_ver}, duplicate={matched}, 建议保留新版={newer_hint}"
                results.append(show("T01 版本去重与新旧判断", ok, detail))
                diff = client.get(f"{base}/api/papers/{v7_id}/diff/{d1['paper_id']}", headers=auth)
                diff.raise_for_status()
                diff_data = diff.json()
                diff_ok = bool(diff_data.get("old_filename") and diff_data.get("new_filename") and diff_data.get("summary"))
                results.append(show("T01 版本差异接口", diff_ok, f"{diff_data.get('summary', '')[:180]}"))
            except Exception as e:
                results.append(show("T01 版本去重与差异", False, str(e)))
        else:
            print("[SKIP] T01：请提供 --v7 和 --v1")

        # T02：扫描 PDF OCR
        if args.scan:
            try:
                if not health.get("ocr_ready"):
                    raise RuntimeError("后端 health 显示 ocr_ready=false；先安装/启用 rapidocr-onnxruntime")
                import time
                print("[RUN] T02 开始上传扫描版 PDF...", flush=True)
                t0 = time.perf_counter()

                data = upload(args.scan)

                print(
                    f"[DONE] T02 上传完成，耗时 {time.perf_counter()-t0:.1f} 秒",
                    flush=True
                )
                meta = data.get("metadata") or {}
                ocr_pages = meta.get("ocr_pages") or []
                first = client.get(f"{base}/api/papers/{data['paper_id']}/pages?start_page=1&page_size=5", headers=auth)
                first.raise_for_status()
                pages = first.json().get("pages") or []
                ocr_text = [x for x in pages if x.get("method") == "ocr" and len((x.get("text") or "").strip()) >= 20]
                ok = bool(ocr_pages and ocr_text)
                results.append(show("T02 扫描 PDF OCR", ok, f"OCR页数={len(ocr_pages)}, 前5页有效OCR文本={len(ocr_text)}"))
            except Exception as e:
                results.append(show("T02 扫描 PDF OCR", False, str(e)))
        else:
            print("[SKIP] T02：请提供 --scan")

        # T03：541 页长文档与分页加载
        book_id = None
        if args.book:
            try:
                start = time.time()
                data = upload(args.book)
                book_id = data["paper_id"]
                page_count = int(data.get("page_count", 0))
                if page_count != 541:
                    raise AssertionError(f"解析页数应为541，实际为{page_count}")
                seen = []
                for page_start in range(1, 542, 20):
                    r = client.get(
                        f"{base}/api/papers/{book_id}/pages",
                        headers=auth,
                        params={"start_page": page_start, "page_size": 20},
                    )
                    r.raise_for_status()
                    seen.extend(x.get("page") for x in (r.json().get("pages") or []))
                ok = seen == list(range(1, 542))
                elapsed = round(time.time() - start, 1)
                results.append(show("T03 541页解析与分页完整性", ok, f"页数={page_count}, 实际读取={len(seen)}, 用时={elapsed}s"))
            except Exception as e:
                results.append(show("T03 541页解析与分页完整性", False, str(e)))
        else:
            print("[SKIP] T03：请提供 --book")

        # T04：RAG 细粒度事实检索与页码溯源（需要 v7 已上传）
        if v7_id:
            cases = [
                ("Adam beta参数", "What beta1 and beta2 values were used for the Adam optimizer?", ["0.9", "0.98"]),
                ("训练GPU型号", "How many GPUs and what GPU model were used to train the Transformer Base model?", ["P100"]),
                ("BLEU实验结果", "What BLEU score did Transformer big achieve on English-to-German, and what was the previous best result?", ["28.4"]),
            ]
            for label, question, expected in cases:
                try:
                    r = client.post(
                        f"{base}/api/qa/search",
                        headers=auth,
                        json={"question": question, "paper_ids": [v7_id], "top_k": 7},
                    )
                    r.raise_for_status()
                    sources = r.json().get("sources") or []
                    good = [s for s in sources if s.get("page") and s.get("filename") and len((s.get("quote") or "").strip()) >= 30]
                    found = [token for token in expected if any(token.lower() in s.get("quote", "").lower() for s in good)]
                    ok = bool(good and len(found) == len(expected))
                    results.append(show(f"T04 {label}", ok, f"证据数={len(sources)}, 命中预期词={found}, 页码/原文锚点={'完整' if good else '缺失'}"))
                    for s in good[:2]:
                        print(f"      {s.get('filename')} p.{s.get('page')}: {s.get('quote','')[:220].replace(chr(10), ' ')}")
                except Exception as e:
                    results.append(show(f"T04 {label}", False, str(e)))
        else:
            print("[SKIP] T04：需要 T01 成功上传 v7 后才能测单篇 RAG")

        # T05：无证据问题必须拒答
        if paper_ids:
            try:
                nonsense = "What is the measured magnetic permeability of the alien crystal ZXQ-99172 in Figure 99?"
                s = client.post(
                    f"{base}/api/qa/search",
                    headers=auth,
                    json={"question": nonsense, "paper_ids": paper_ids, "top_k": 7},
                )
                s.raise_for_status()
                no_sources = not (s.json().get("sources") or [])
                stream = client.post(
                    f"{base}/api/qa/stream",
                    headers=auth,
                    json={"paper_id": paper_ids[0], "question": nonsense},
                )
                stream.raise_for_status()
                refused = ("没有检索到足够" in stream.text) or ("证据不足" in stream.text)
                ok = no_sources and refused
                results.append(show("T05 无证据问题拒答", ok, f"检索为空={no_sources}, 回答明确拒答={refused}"))
            except Exception as e:
                results.append(show("T05 无证据问题拒答", False, str(e)))
        else:
            print("[SKIP] T05：至少成功上传一篇 PDF 后才能测拒答")

    except Exception as e:
        print(f"[FATAL] 测试初始化失败：{e}")
        sys.exit(2)
    finally:
        client.close()

    ran = len(results)
    passed = sum(results)
    failed = ran - passed
    print("\n========== T01-T05 汇总 ==========")
    print(f"已执行：{ran} | PASS：{passed} | FAIL：{failed}")
    print("注意：T04 为检索证据锚点验收；答案是否正确仍需人工对照 PDF 原文。")
    print("注意：T05 只验证无证据拒答，不等于证明证据验证 Agent 已独立完成事实核验。")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
