"use client";




import { Suspense, useEffect, useState, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { CheckCircle2, GitCompare, Loader2, Send, X, FileText, BookOpen, Layers, } from "lucide-react";
import { apiFetch, consumeSSE } from "@/lib/api";

const cleanMarkdown = (md: string) => {
  if (!md) return "";
  return md
    .replace(/^### (.*$)/gim, '【$1】')
    .replace(/^## (.*$)/gim, '【$1】')
    .replace(/^# (.*$)/gim, '【$1】')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/^> (.*$)/gim, '💡 $1')
    .replace(/^- (.*$)/gim, '● $1')
    .replace(/^\|?(.*?)\|$/gim, (match) => match.replace(/\|/g, ' '))
    .replace(/^---$/gim, '————————');
};

type Step = { agent: string; status?: string };
type Paper = { id: string; filename: string; title?: string };

function CompareContent() {
  const params = useSearchParams();
  const initialIds = (params.get("ids") || "").split(",").filter(Boolean).slice(0, 12);

  const [selectedIds, setSelectedIds] = useState<string[]>(initialIds);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // 抽屉状态
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const [question, setQuestion] = useState("请对比这些论文的核心创新点、研究方法、实验结果，以及它们之间的技术演进关系。");
  const [steps, setSteps] = useState<Step[]>([]);
  const [answer, setAnswer] = useState("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [sources, setSources] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchPapers = async () => {
      try {
        const res = await apiFetch("/api/papers");
        if (res.ok) {
          const data = await res.json();
          setPapers(Array.isArray(data) ? data : []);
        }
      } catch (e) {
        console.error("加载文献列表失败", e);
      }
    };
    void fetchPapers();
  }, []);

  const togglePaper = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(0, 12)
    );
  };

  const run = async () => {
    if (selectedIds.length < 2 || !question.trim() || running) return;
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setRunning(true); setSteps([]); setAnswer(""); setSources([]); setError(""); setIsDrawerOpen(false);

    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }

    try {
      const res = await apiFetch("/api/qa/compare", {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ paper_ids: selectedIds, question }),
      });
      await consumeSSE(res, (data) => {
        if (data.agent) {
          setSteps((old) => {
            const index = old.findIndex((x) => x.agent === data.agent);
            if (index >= 0) return old.map((x, i) => i === index ? { ...x, ...data } : x);
            return [...old, data];
          });
        }
        if (data.text) setAnswer((old) => old + data.text);
        if (data.sources) setSources(data.sources);
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : "跨文献分析失败";
      setError(message);
      if (message.includes("登录")) window.dispatchEvent(new Event("open-login-modal"));
    } finally { setRunning(false); }
  };

  const selectedPapers = selectedIds.map(id => papers.find(p => p.id === id) || { id, filename: "加载中...", title: "" });
  const unselectedPapers = papers.filter(p => !selectedIds.includes(p.id));

  return (
    <div className="flex h-[calc(100vh-64px)] w-full overflow-hidden bg-slate-50 font-sans">

      {/* 左侧：控制面板 */}
      <div className="flex w-[340px] shrink-0 flex-col border-r border-slate-200 bg-white shadow-[4px_0_24px_rgba(0,0,0,0.02)] z-10">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b border-slate-100 px-5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-sm">
            <GitCompare className="h-4 w-4" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-800">跨文献分析</h1>
            <p className="text-[10px] text-slate-400 font-medium">MULTI-PAPER REASONING</p>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-xs font-bold text-slate-500">已选文献 ({selectedIds.length}/12)</h2>
              <div className="relative">
                <button onClick={() => setShowDropdown(!showDropdown)} disabled={selectedIds.length >= 12 || unselectedPapers.length === 0} className="text-xs text-indigo-600 hover:text-indigo-800 font-medium disabled:opacity-50">+ 添加</button>
                {showDropdown && unselectedPapers.length > 0 && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => { setShowDropdown(false); setSearchQuery(""); }}></div>
                  <div className="absolute right-0 top-full z-20 mt-1 w-72 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl flex flex-col">
                    {/* 搜索框 */}
                    <div className="p-2 border-b border-slate-100 bg-slate-50/80 sticky top-0 z-10">
                      <input
                        type="text"
                        autoFocus
                        placeholder="搜索文献..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400 placeholder:text-slate-400"
                      />
                    </div>

                    {/* 列表区 */}
                    <div className="max-h-60 overflow-y-auto py-1">
                      {unselectedPapers
                        .filter(p => (p.title || p.filename).toLowerCase().includes(searchQuery.toLowerCase()))
                        .map(p => (
                          <button
                            key={p.id}
                            onClick={() => { togglePaper(p.id); setShowDropdown(false); setSearchQuery(""); }}
                            className="block w-full truncate px-4 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50 hover:text-indigo-900 transition-colors"
                            title={p.title || p.filename}
                          >
                            {p.title || p.filename}
                          </button>
                      ))}

                      {unselectedPapers.filter(p => (p.title || p.filename).toLowerCase().includes(searchQuery.toLowerCase())).length === 0 && (
                        <p className="px-4 py-4 text-center text-xs text-slate-400">未找到匹配的文献</p>
                      )}
                    </div>
                  </div>
                </>
              )}
              </div>
            </div>
            <div className="space-y-1.5">
              {selectedPapers.map((p, idx) => (
                <div key={`${p.id}-${idx}`} className="group flex items-center justify-between gap-2 rounded-md bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700">
                  <span className="truncate flex-1" title={p.title || p.filename}>{p.title || p.filename}</span>
                  <button onClick={() => togglePaper(p.id)} className="text-slate-400 hover:text-rose-500"><X className="h-3 w-3" /></button>
                </div>
              ))}
            </div>
          </section>

          {steps.length > 0 && (
            <section>
              <h2 className="text-xs font-bold text-slate-500 mb-2">执行进度</h2>
              <div className="space-y-2">
                {steps.map((step, index) => (
                  <div key={`${step.agent}-${index}`} className="flex items-center gap-2 text-xs text-slate-600">
                    {step.status === "running" ? <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" /> : <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />}
                    <span>{step.agent}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {error && <div className="rounded-lg bg-rose-50 p-2.5 text-xs text-rose-700">{error}</div>}
        </div>

        <div className="shrink-0 bg-white p-4 pb-6">
          <div className="relative flex flex-col rounded-2xl border border-slate-200 bg-slate-50 transition-all focus-within:border-indigo-300 focus-within:bg-white focus-within:shadow-[0_4px_20px_-4px_rgba(79,70,229,0.15)] focus-within:ring-1 focus-within:ring-indigo-300">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if(e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void run(); } }}
              disabled={running}
              rows={3}
              className="w-full resize-none bg-transparent px-4 pt-3 pb-1 text-sm leading-6 text-slate-700 outline-none placeholder:text-slate-400 disabled:opacity-50 scrollbar-thin"
              placeholder="输入分析指令，Shift+Enter 换行..."
            />
            <div className="flex items-center justify-between px-3 pb-2">
              <span className="text-[10px] text-slate-400 font-medium">支持多篇联合推理</span>
              <button
                onClick={() => void run()}
                disabled={running || selectedIds.length < 2}
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-sm transition-all hover:bg-indigo-700 hover:shadow disabled:bg-slate-300 disabled:shadow-none"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4 ml-0.5" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 右侧：主阅读区 */}
      <div className="flex-1 min-w-0 flex flex-col relative bg-slate-50/50">
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-8 py-8 md:px-12 md:py-10 scroll-smooth">
          <div className="mx-auto max-w-4xl space-y-6 pb-20">

            {!answer && !running && (
              <div className="flex h-[60vh] flex-col items-center justify-center text-slate-400">
                <GitCompare className="h-16 w-16 text-slate-200 mb-4" />
                <p className="text-sm font-medium text-slate-500">报告画布已就绪</p>
              </div>
            )}

            {running && !answer && (
              <div className="space-y-4 animate-pulse mt-10">
                <div className="h-8 bg-slate-200 rounded-md w-1/3"></div>
                <div className="h-4 bg-slate-200 rounded-md w-full"></div>
                <div className="h-4 bg-slate-200 rounded-md w-5/6"></div>
                <div className="h-4 bg-slate-200 rounded-md w-4/6"></div>
              </div>
            )}

            {answer && (
              <div className="rounded-2xl border border-slate-200 bg-white p-8 md:p-12 shadow-sm">
                <div className="flex items-center gap-3 mb-8 border-b border-slate-100 pb-4">
                  <FileText className="h-6 w-6 text-indigo-600" />
                  <h2 className="text-xl font-bold text-slate-800">跨文献综合分析报告</h2>
                </div>
                <div className="whitespace-pre-wrap text-[15px] leading-8 text-slate-700">
                  {cleanMarkdown(answer)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 悬浮证据按钮 */}
        {sources.length > 0 && (
          <button
            onClick={() => setIsDrawerOpen(true)}
            className="absolute bottom-8 right-8 z-20 flex items-center gap-2 rounded-full bg-slate-900 px-5 py-3 text-sm font-medium text-white shadow-xl transition-all hover:bg-slate-800 hover:scale-105 active:scale-95"
          >
            <Layers className="h-4 w-4 text-emerald-400" />
            查看原文证据 ({sources.length})
          </button>
        )}
      </div>

      {/* 右侧滑出抽屉（遮罩层） */}
      {isDrawerOpen && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* 背景遮罩 */}
          <div className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm transition-opacity" onClick={() => setIsDrawerOpen(false)}></div>

          {/* 抽屉面板 */}
          <div className="relative flex h-full w-full max-w-[380px] flex-col bg-white shadow-2xl animate-in slide-in-from-right duration-300">
            <header className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 px-6">
              <div className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-emerald-600" />
                <h2 className="text-lg font-bold text-slate-800">原文证据溯源</h2>
              </div>
              <button onClick={() => setIsDrawerOpen(false)} className="rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors">
                <X className="h-5 w-5" />
              </button>
            </header>

            <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
              {sources.map((source, i) => (
                <div key={source.chunk_id || i} className="group rounded-xl border border-slate-200 bg-white p-3 shadow-sm hover:border-indigo-300 hover:shadow-md transition-all">
                  <div className="flex items-center justify-between mb-1.5 border-b border-slate-50 pb-1.5">
                    <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-bold text-indigo-700">{source.evidence_id}</span>
                    <span className="text-[10px] text-slate-400 truncate max-w-[150px]">
                      <BookOpen className="inline h-3 w-3 mr-1 relative -top-[1px]" />
                      {source.filename} · P{source.page}
                    </span>
                  </div>
                  <p className="text-[11px] leading-[1.6] text-slate-600 line-clamp-5 group-hover:line-clamp-none transition-all duration-300">
                    {source.quote}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default function ComparePage() {
  return <Suspense fallback={<div className="p-10 text-center text-sm text-slate-500">加载工作台…</div>}><CompareContent /></Suspense>;
}
