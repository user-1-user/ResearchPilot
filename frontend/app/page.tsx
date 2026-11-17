"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity, ArrowRight, BookOpen, BrainCircuit, CheckCircle2, FileText,
  Lightbulb, Loader2, Network, PenTool, Search, Send, Sparkles, Wrench, Eye, RefreshCw, ChevronRight
} from "lucide-react";
import { apiFetch, consumeSSE } from "@/lib/api";

interface AgentStep {
  type: "thought" | "action" | "observation" | "error";
  content: string;
}

const shortcuts = [
  { title: "文献库", desc: "上传论文、去重与证据检索", href: "/library", icon: BookOpen },
  { title: "引用图谱", desc: "发现领域基石论文", href: "/graph", icon: Network },
  { title: "科研 Copilot", desc: "生成框架、分析实验数据", href: "/copilot", icon: PenTool },
];

const suggestionsPool = [
  "统计我的文献库，并告诉我总页数",
  "对比库中论文的核心创新与技术演进",
  "提取论文中的 Future Work 并归纳研究空白",
  "根据我的文献库生成一份有真实引用依据的论文框架",
  "帮我写一份关于 Transformer 架构演变的技术综述",
  "去 arXiv 检索最近关于大模型 Agent 的最新论文",
  "验证『利用扩散模型提升大模型推理能力』这个 Idea 的新颖性",
  "分析 BERT 和 GPT-3 在预训练架构上的核心区别",
  "总结我文献库中所有论文的局限性 (Limitations)",
  "帮我把实验产出的 CSV 数据生成对应的学术图表和 Caption",
];

const capabilities = [
  "本地文献统计与混合检索",
  "跨文献证据对比",
  "Future Work 与引用图谱",
  "arXiv 论文检索",
  "有证据约束的论文框架生成"
];

export default function Dashboard() {
  const [input, setInput] = useState("帮我查一下库里有多少篇论文？");
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [answer, setAnswer] = useState("");
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<{ paper_count: number; page_count: number; chunk_count: number } | null>(null);
  const [notice, setNotice] = useState("");
  const [currentSuggestions, setCurrentSuggestions] = useState<string[]>(suggestionsPool.slice(0, 1));

  const refreshSuggestions = () => {
    // 这里就不查库了，直接用通用的灵感池随机换一批
    const shuffled = [...suggestionsPool].sort(() => 0.5 - Math.random());
    setCurrentSuggestions(shuffled.slice(0, 3));
  };


  const loadStats = useCallback(async () => {
    try {
      const res = await apiFetch("/api/papers");
      if (!res.ok) return;
      const papers = await res.json();
      setStats({
        paper_count: Array.isArray(papers) ? papers.length : 0,
        page_count: Array.isArray(papers) ? papers.reduce((n: number, p: any) => n + (p.page_count || 0), 0) : 0,
        chunk_count: 0,
      });
    } catch {
      // The dashboard remains usable while the backend is offline.
    }
  }, []);

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void loadStats(); }, [loadStats]);

  const runAgent = async () => {
    const message = input.trim();
    if (!message || running) return;
    if (!localStorage.getItem("yantu_token")) {
      window.dispatchEvent(new Event("open-login-modal"));
      return;
    }
    setRunning(true);
    setSteps([]);
    setAnswer("");
    setNotice("");
    try {
      const response = await apiFetch("/api/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message }),
      });
      if (response.status === 401) {
        window.dispatchEvent(new Event("open-login-modal"));
        throw new Error("登录状态已失效，请重新登录。");
      }
      await consumeSSE(response, (data) => {
        if (["thought", "action", "observation", "error"].includes(data.type)) {
          setSteps((prev) => [...prev, { type: data.type, content: String(data.content ?? "") }]);
        }
        if (data.type === "text") setAnswer((prev) => prev + String(data.content ?? ""));
      });
      await loadStats();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Agent 执行失败，请检查后端服务。");
    } finally {
      setRunning(false);
    }
  };

  const stepIcon = (type: AgentStep["type"]) => {
    if (type === "thought") return <BrainCircuit className="h-4 w-4 text-violet-600" />;
    if (type === "action") return <Wrench className="h-4 w-4 text-amber-600" />;
    if (type === "observation") return <Eye className="h-4 w-4 text-emerald-600" />;
    return <Activity className="h-4 w-4 text-rose-600" />;
  };

  const cleanMarkdown = (md: string) => {
    if (!md) return "";
    return md
      .replace(/### (.*)/g, '【$1】')
      .replace(/## (.*)/g, '【$1】')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/^- /gm, '● ');
  };

  return (
    // 外层使用 min-h-screen + pb-32，确保内容自然延伸，底部留出输入框空间
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 pb-32">
      <div className="mx-auto max-w-[1600px] space-y-6 p-6 md:p-8">

        {/* 顶部 Banner */}
              {/* 顶部 Banner */}
      <div className="relative">

        <section className="relative overflow-hidden rounded-[2.5rem] bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-900 p-8 text-white shadow-xl md:p-10 relative z-10">
        <div className="pointer-events-none absolute -right-20 -top-24 h-80 w-80 rounded-full bg-blue-400/15 blur-3xl" />
        <div className="relative flex flex-col justify-between gap-8 md:flex-row md:items-end">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-300/25 bg-white/10 px-3 py-1 text-xs text-blue-100">
              <Sparkles className="h-3.5 w-3.5" /> ResearchPilot · AI for Science
            </div>
            <h1 className="text-3xl font-bold tracking-tight md:text-4xl">让科研，从检索走向发现</h1>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-blue-100/80 md:text-base">
              多智能体协同完成论文解析、证据检索、跨文献推理、研究机会探索与学术写作。每个结论都尽可能回到原文证据。
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 md:min-w-64">
            <div className="p-2 transition-colors hover:bg-white/5 rounded-2xl">
              <FileText className="mb-3 h-5 w-5 text-blue-200" />
              <div className="text-2xl font-semibold">{stats?.paper_count ?? "—"}</div>
              <div className="mt-1 text-xs text-blue-100/70">已入库论文</div>
            </div>
            <div className="p-2 transition-colors hover:bg-white/5 rounded-2xl">
              <BookOpen className="mb-3 h-5 w-5 text-blue-200" />
              <div className="text-2xl font-semibold">{stats?.page_count ?? "—"}</div>
              <div className="mt-1 text-xs text-blue-100/70">已解析页数</div>
            </div>
          </div>
        </div>
      </section>
      </div>

      {/* 快捷入口 */}
      <section className="flex flex-wrap justify-center gap-4 md:gap-6">
        {shortcuts.map((item) => {
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href} className="group w-full max-w-[340px] rounded-2xl border border-slate-200 bg-white shadow-sm p-5 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-500/10">
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-100 to-blue-50/50 text-blue-600 shadow-sm ring-1 ring-blue-100/50 transition-all duration-300 group-hover:from-blue-600 group-hover:to-blue-500 group-hover:text-white group-hover:shadow-md">
                <Icon className="h-5 w-5" />
              </div>
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-slate-900">{item.title}</h2>
                <div className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-100 bg-white shadow-sm transition-all duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-1 group-hover:border-blue-200 group-hover:bg-blue-50 group-hover:shadow-md">
                    <ArrowRight className="h-4 w-4 text-slate-400 transition-colors group-hover:text-blue-600" />
                  </div>
              </div>
              <p className="mt-1 text-sm text-slate-500">{item.desc}</p>
            </Link>
          );
        })}
      </section>

{/* 主控 Agent 工作区 (自然撑开高度) */}
        <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm flex flex-col">
          <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-5 bg-white">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 text-blue-600"><BrainCircuit className="h-5 w-5" /></div>
            <div>
              <h2 className="font-semibold text-slate-800">科研任务总控 Agent</h2>
              <p className="text-xs text-slate-500">Task Planning · Tool Calling · Multi-Agent Workflow</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Agent 已就绪 · 安全护栏开启
          </div>
        </div>

          <div className="p-6 space-y-4">
            {steps.length === 0 && !answer && (
            <div className="flex h-full flex-col items-center justify-center text-center max-w-4xl mx-auto py-8">
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-50 text-blue-600"><Sparkles className="h-8 w-8" /></div>
              <p className="text-xl font-bold text-slate-800">告诉 Agent 你想完成什么科研任务</p>
              <p className="mt-2 text-sm text-slate-500">系统将规划步骤、调用真实工具、检索证据并整理结果。</p>

              <div className="mt-10 flex w-full justify-center text-left">
                {/* 灵感推荐 */}
                <div className="flex w-full max-w-2xl flex-col rounded-2xl border border-slate-200/60 bg-white p-5 shadow-sm">
                  <div className="mb-4 flex items-center justify-between text-xs font-bold text-slate-600">
                    <div className="flex items-center gap-1.5"><Lightbulb className="h-4 w-4 text-amber-500" /> 灵感推荐</div>
                    <button onClick={refreshSuggestions} className="p-1 text-slate-400 hover:text-blue-500 transition-colors" title="换一批">
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="flex flex-1 flex-col gap-2">
                    {currentSuggestions.map((suggestion) => (
                      <button key={suggestion} onClick={() => setInput(suggestion)} className="group flex w-full flex-1 items-center justify-between rounded-xl border border-transparent bg-slate-50 px-4 py-3 text-left text-sm text-slate-600 transition-all hover:border-slate-200 hover:bg-white hover:text-blue-700 hover:shadow-sm">
                        <span className="line-clamp-2">{suggestion}</span>
                        <ArrowRight className="h-4 w-4 shrink-0 text-slate-300 transition-transform group-hover:translate-x-1 group-hover:text-blue-500" />
                      </button>
                    ))}
                  </div>
                </div>

                {/* 系统能力 */}

              </div>
            </div>
          )}

          {steps.length > 0 && (
              <div className="space-y-3">
                {steps.map((step, index) => (
                  <div key={`${index}-${step.type}`} className="flex gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                    <div className="mt-0.5">{stepIcon(step.type)}</div>
                    <div className="min-w-0 flex-1">
                      <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">{step.type}</div>
                      {step.type === 'observation' ? (
                        <details className="group">
                          <summary className="cursor-pointer text-xs font-semibold text-blue-600 hover:text-blue-700 outline-none select-none flex items-center gap-1.5">
                            <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
                            查看工具返回的原始数据 (JSON)
                          </summary>
                          <div className="mt-2 whitespace-pre-wrap rounded-lg bg-slate-50 p-3 font-mono text-[11px] leading-5 text-slate-600 border border-slate-200">
                            {step.content}
                          </div>
                        </details>
                      ) : (
                        <p className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">{step.content}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {answer && (
              <div className="rounded-xl border border-blue-200 bg-blue-50/50 p-5 shadow-sm">
                <div className="mb-4 flex items-center gap-2 text-sm font-bold text-blue-800"><CheckCircle2 className="h-5 w-5" /> AGENT RESULT</div>
                <div className="whitespace-pre-wrap text-sm leading-7 text-slate-800">{cleanMarkdown(answer)}</div>
              </div>
            )}

            {notice && <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{notice}</div>}
            {running && <div className="flex items-center gap-2 text-sm text-blue-700 p-2"><Loader2 className="h-4 w-4 animate-spin" /> Agent 正在执行任务…</div>}
          </div>
        </section>
      </div>

      {/* 底部渐变融合区 */}
      <div className="fixed bottom-0 left-0 right-0 md:left-64 z-30 bg-gradient-to-t from-slate-50 via-slate-50/95 to-transparent pb-8 pt-20 pointer-events-none">
        <div className="mx-auto w-full max-w-7xl px-6 md:px-8 pointer-events-auto">

          <div className="relative flex items-center gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-[0_8px_30px_rgb(0,0,0,0.08)] focus-within:border-blue-500 focus-within:ring-4 focus-within:ring-blue-500/10 transition-all">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void runAgent()}
              placeholder="输入你想完成的科研任务..."
              disabled={running}
              className="min-w-0 flex-1 bg-transparent px-4 py-2 text-base text-slate-800 outline-none disabled:opacity-50"
            />
            <button
              onClick={() => void runAgent()}
              disabled={running || !input.trim()}
              className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-blue-600 px-8 py-3 text-sm font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:bg-blue-700 disabled:translate-y-0 disabled:opacity-50"
            >
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} 执行
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}
