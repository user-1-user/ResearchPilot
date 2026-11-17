"use client";




import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { AlertTriangle, BookOpen, Check, FileText, GitCompare, Loader2, Trash2, Upload, Search, X } from "lucide-react";
import { apiFetch } from "@/lib/api";

type Paper = { id: string; filename: string; page_count: number; title?: string; created_at?: string };
type Duplicate = { newPaperId: string; filename: string; candidate: { paper_id: string; filename: string; similarity: number; suggestion: string } };

export default function LibraryPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ isUploading: false, currentFile: "", currentIndex: 0, total: 0 });
  const [mockProgress, setMockProgress] = useState(0);
  const [error, setError] = useState("");
  const [duplicateQueue, setDuplicateQueue] = useState<Duplicate[]>([]);
  const [diffResult, setDiffResult] = useState<{ summary: string; diff_excerpt: string[] } | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // 当前需要处理的重复论文是队列中的第一篇。
  const duplicate = duplicateQueue[0] ?? null;

  const fetchPapers = useCallback(async () => {
    try {
      const res = await apiFetch("/api/papers");
      if (res.status === 401) return;
      if (!res.ok) throw new Error("文献列表加载失败");
      const data = await res.json();
      setPapers(Array.isArray(data) ? data : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "文献列表加载失败");
    }
  }, []);

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void fetchPapers(); }, [fetchPapers]);

  // 模拟平滑进度条
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (uploadStatus.isUploading) {
      setMockProgress(5); // 初始给个底
      interval = setInterval(() => {
        setMockProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return 95;
          }
          // 越往后涨得越慢
          const step = Math.max(1, Math.floor((95 - prev) / 10));
          return prev + step;
        });
      }, 400);
    } else {
      setMockProgress(100);
      setTimeout(() => setMockProgress(0), 500); // 结束后重置
    }
    return () => clearInterval(interval);
  }, [uploadStatus.isUploading]);

  const chooseUpload = () => {
    if (duplicateQueue.length > 0) {
      setError("请先处理当前待确认的重复论文，再开始新一批上传。");
      return;
    }

    if (!localStorage.getItem("yantu_token")) {
      window.dispatchEvent(new Event("open-login-modal"));
      return;
    }

    inputRef.current?.click();
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);

    // 清空 input，允许用户再次选择同一批文件。
    event.target.value = "";

    if (!files.length) return;

    setLoading(true);
    setError("");
    setUploadStatus({ isUploading: true, currentFile: "", currentIndex: 0, total: files.length });

    // 收集本批次所有疑似重复论文。
    const foundDuplicates: Duplicate[] = [];

    // 收集完全相同的文件和上传失败信息。
    const identicalFiles: string[] = [];
    const failedFiles: string[] = [];

    try {
      // 逐篇上传。某一篇失败或疑似重复，都不终止后续文件。
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadStatus(prev => ({ ...prev, currentFile: file.name, currentIndex: i + 1 }));
        try {
          const form = new FormData();
          form.append("file", file);

          const res = await apiFetch("/api/papers/upload", {
            method: "POST",
            body: form,
          });

          const data = await res.json().catch(() => ({}));

          if (!res.ok) {
            const detail =
              typeof data.detail === "string"
                ? data.detail
                : "服务器返回了无法识别的错误";

            failedFiles.push(`${file.name}：${detail}`);
            continue;
          }

          // 完全相同的文件已存在，不需要重复入库。
          if (data.already_exists) {
            identicalFiles.push(file.name);
            continue;
          }

          // 疑似重复：加入队列，不再 break。
          if (
            Array.isArray(data.duplicates) &&
            data.duplicates.length > 0 &&
            data.paper_id
          ) {
            foundDuplicates.push({
              newPaperId: data.paper_id,
              filename: file.name,
              candidate: data.duplicates[0],
            });
          }
        } catch (fileError) {
          failedFiles.push(
            `${file.name}：${
              fileError instanceof Error
                ? fileError.message
                : "未知上传错误"
            }`
          );

          // 继续上传下一篇。
          continue;
        }
      }

      // 所有文件处理完成后，再统一设置待处理重复论文队列。
      setDuplicateQueue(foundDuplicates);
      setUploadStatus({ isUploading: false, currentFile: "", currentIndex: 0, total: 0 });

      // 汇总完全重复和失败的文件。
      const messages: string[] = [];

      if (identicalFiles.length > 0) {
        messages.push(
          `以下文件与库中已有文件完全相同，已跳过：${identicalFiles.join("、")}`
        );
      }

      if (failedFiles.length > 0) {
        messages.push(
          `以下文件上传失败：${failedFiles.join("；")}`
        );
      }

      setError(messages.join("\n"));

      // 刷新文献列表，显示本批次成功入库的文件。
      await fetchPapers();
    } finally {
      setLoading(false);
    }
  };

  const loadVersionDiff = async () => {
    if (!duplicate) return;
    setDiffLoading(true);
    try {
      const res = await apiFetch(`/api/papers/${duplicate.newPaperId}/diff/${duplicate.candidate.paper_id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "版本差异分析失败");
      setDiffResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "版本差异分析失败");
    } finally {
      setDiffLoading(false);
    }
  };

  const resolveDuplicate = async (action: "keep_new" | "keep_existing" | "keep_both") => {
    if (!duplicate) return;

    try {
      const res = await apiFetch(
        `/api/papers/${duplicate.newPaperId}/resolve-duplicate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            candidate_id: duplicate.candidate.paper_id,
            action,
          }),
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "版本处理失败");
      }

      // 当前重复论文处理完成，从队列中移除第一项。
      // 下一篇重复论文会自动显示。
      setDuplicateQueue((previous) => previous.slice(1));

      setDiffResult(null);
      setSelected([]);

      await fetchPapers();

      setError(data.message || "版本处理完成");
    } catch (e) {
      setError(e instanceof Error ? e.message : "版本处理失败");
    }
  };

  const removePaper = async (paper: Paper) => {
    if (!window.confirm(`确定删除《${paper.title || paper.filename}》吗？`)) return;
    const res = await apiFetch(`/api/papers/${paper.id}`, { method: "DELETE" });
    if (!res.ok) { setError("删除失败"); return; }
    setSelected((prev) => prev.filter((id) => id !== paper.id)); await fetchPapers();
  };

  const toggle = (id: string) => setSelected((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);

  const compare = () => {
    if (selected.length < 2) { setError("请至少选择两篇论文进行对比。"); return; }
    router.push(`/compare?ids=${selected.join(",")}`);
  };

  // 提取过滤逻辑，修复 JSX 语法错误
  const filteredPapers = papers.filter(paper => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return true;
    return (paper.title || "").toLowerCase().includes(q) || paper.filename.toLowerCase().includes(q);
  });

  return (
    <div className="mx-auto min-h-full max-w-6xl bg-slate-50 p-6 md:p-8">
      {/* 全屏解析进度弹窗 */}
      {loading && uploadStatus.isUploading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm">
          <div className="w-[400px] overflow-hidden rounded-2xl bg-white shadow-2xl ring-1 ring-slate-900/5">
            <div className="flex flex-col items-center gap-4 px-6 py-10">
              <div className="relative flex h-20 w-20 items-center justify-center">
                <div className="absolute inset-0 animate-ping rounded-full bg-blue-100 opacity-75"></div>
                <div className="relative flex h-14 w-14 items-center justify-center rounded-full bg-blue-600 shadow-lg">
                  <Loader2 className="h-7 w-7 animate-spin text-white" />
                </div>
              </div>
              <div className="text-center">
                <h3 className="text-xl font-bold text-slate-900">AI 正在深度解析</h3>
                <p className="mt-2 text-sm text-slate-500 font-medium">
                  {mockProgress < 20 ? "正在提取 PDF 文本与版面结构..." :
                   mockProgress < 50 ? "正在识别表格、公式与参考文献..." :
                   mockProgress < 75 ? "正在进行语义分块与数据清洗..." :
                   mockProgress < 95 ? "正在调用 BGE-M3 生成稠密向量索引..." :
                   "正在写入本地知识库，即将完成..."}
                </p>
                {uploadStatus.currentFile && (
                  <p className="mt-1 text-xs text-slate-400 truncate max-w-[300px] mx-auto" title={uploadStatus.currentFile}>
                    📄 {uploadStatus.currentFile}
                  </p>
                )}
              </div>
              <div className="w-full rounded-full bg-slate-100 h-2.5 mt-4 overflow-hidden">
                <div
                  className="h-2.5 rounded-full bg-blue-600 transition-all duration-700 ease-out"
                  style={{ width: `${mockProgress}%` }}
                ></div>
              </div>
              <div className="w-full flex justify-between text-xs font-semibold text-slate-400 mt-1">
                <span>解析进度</span>
                <span>{mockProgress}%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600"><BookOpen className="h-4 w-4" /> Knowledge Base</div>
          <h1 className="text-3xl font-bold text-slate-900">文献知识库</h1>
          <p className="mt-2 text-sm text-slate-500">语义去重、版本管理、分块索引与可追溯问答。</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {selected.length > 0 && <button onClick={compare} className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700"><GitCompare className="h-4 w-4" /> 对比 {selected.length} 篇</button>}
          <button onClick={chooseUpload} disabled={loading} className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />} {loading ? "解析中…" : "上传 PDF"}</button>
          <input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple className="hidden" onChange={handleUpload} />
        </div>
      </div>

      {error && <div className="mb-5 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">{error}<button onClick={() => setError("")} className="ml-3 font-semibold underline">关闭</button></div>}

      {duplicate && (
        <div className="mb-6 rounded-2xl border border-amber-300 bg-amber-50 p-5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
            <div className="min-w-0 flex-1">
              <h2 className="font-semibold text-amber-950">发现疑似重复论文 / 不同版本</h2>
              <p className="mt-1 text-sm text-amber-900">新文件：{duplicate.filename}</p>
              <p className="text-sm text-amber-900">库中版本：{duplicate.candidate.filename} · 相似度 {(duplicate.candidate.similarity * 100).toFixed(1)}%</p>
              <p className="mt-2 text-sm text-amber-800">{duplicate.candidate.suggestion}</p>
              <button onClick={() => void loadVersionDiff()} disabled={diffLoading} className="mt-3 inline-flex items-center gap-2 text-sm font-semibold text-amber-900 underline disabled:opacity-50">{diffLoading ? "正在比较…" : "查看两个版本的文本差异"}</button>
              {diffResult && <div className="mt-3 rounded-lg border border-amber-200 bg-white p-3"><p className="text-sm font-medium text-slate-800">{diffResult.summary}</p>{diffResult.diff_excerpt?.length > 0 && <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-600">{diffResult.diff_excerpt.join("\n")}</pre>}</div>}
              <div className="mt-4 flex flex-wrap gap-2">
                <button onClick={() => void resolveDuplicate("keep_existing")} className="rounded-lg border border-amber-400 bg-white px-3 py-2 text-sm text-amber-900 hover:bg-amber-100">保留库中版本</button>
                <button onClick={() => void resolveDuplicate("keep_new")} className="rounded-lg bg-amber-700 px-3 py-2 text-sm font-semibold text-white hover:bg-amber-800">用新版本替换</button>
                <button onClick={() => void resolveDuplicate("keep_both")} className="rounded-lg border border-amber-400 bg-transparent px-3 py-2 text-sm text-amber-900 hover:bg-amber-100">两个版本都保留</button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mb-4 flex items-center justify-between text-sm text-slate-500">
        <span>共 {papers.length} 篇论文 · {papers.reduce((n, p) => n + (p.page_count || 0), 0)} 页</span>
        <button onClick={() => void fetchPapers()} className="hover:text-blue-700">刷新列表</button>
      </div>

      {papers.length === 0 && (
        <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white px-6 py-20 text-center">
          <FileText className="mx-auto h-10 w-10 text-slate-300" />
          <h2 className="mt-4 font-semibold text-slate-700">知识库还没有论文</h2>
          <p className="mt-1 text-sm text-slate-400">上传 PDF 后，系统会自动解析页面、切分文本并建立检索索引。</p>
          <button onClick={chooseUpload} className="mt-5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white">上传第一篇论文</button>
        </div>
      )}

      {papers.length > 0 && (
        <>
          {/* 搜索框 */}
          <div className="mb-4 flex w-full items-center gap-2">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="搜索论文标题或文件名..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full rounded-xl border border-slate-200 bg-white py-2.5 pl-10 pr-10 text-sm outline-none transition-all focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {filteredPapers.map((paper) => {
              const checked = selected.includes(paper.id);
              return (
                <article key={paper.id} className={`rounded-2xl border bg-white p-5 shadow-sm transition ${checked ? "border-indigo-400 ring-2 ring-indigo-100" : "border-slate-200 hover:border-blue-200 hover:shadow-md"}`}>
                  <div className="mb-4 flex items-start justify-between gap-3">
                    <button onClick={() => toggle(paper.id)} aria-label="选择论文" className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md border ${checked ? "border-indigo-600 bg-indigo-600 text-white" : "border-slate-300 text-transparent"}`}><Check className="h-4 w-4" /></button>
                    <button onClick={() => void removePaper(paper)} title="删除论文" className="text-slate-300 hover:text-rose-600"><Trash2 className="h-4 w-4" /></button>
                  </div>
                  <FileText className="mb-3 h-9 w-9 text-blue-600" />
                  <h3 className="line-clamp-2 min-h-12 font-semibold leading-6 text-slate-900" title={paper.title || paper.filename}>{paper.title || paper.filename.replace(/\.pdf$/i, "")}</h3>
                  <p className="mt-2 text-xs text-slate-400">{paper.page_count} 页 · {paper.filename}</p>
                  <Link href={`/reader/${paper.id}`} className="mt-5 block rounded-lg border border-slate-200 py-2 text-center text-sm font-medium text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700">打开论文阅读器</Link>
                </article>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
