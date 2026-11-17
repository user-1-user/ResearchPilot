"use client";




import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { Lightbulb, Loader2, Network, FileText, RefreshCw } from "lucide-react";
import { apiFetch } from "@/lib/api";

type GraphNode = { id: string; name: string; size: number; pagerank?: number; in_degree?: number; out_degree?: number };
type GraphData = { nodes: GraphNode[]; edges: { source: string; target: string }[]; metrics?: { paper_count?: number; citation_edges?: number; top_foundational_papers?: GraphNode[]; isolated_papers?: string[] } };
type FutureItem = { paper_id: string; filename: string; future_work: string[]; future_work_evidence?: { page: number; quote: string }[] };
function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[char];
  });
}

export default function GraphPage() {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [futureWork, setFutureWork] = useState<FutureItem[]>([]);
  const [aiSummary, setAiSummary] = useState<string>("");
  const [futureLoading, setFutureLoading] = useState(false);
  const [error, setError] = useState("");

  const loadGraph = async () => {
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setGraphLoading(true); setError("");
    try {
      const res = await apiFetch("/api/graph/build", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "图谱构建失败");
      setGraph(data);
    } catch (e) { setError(e instanceof Error ? e.message : "图谱构建失败"); }
    finally { setGraphLoading(false); }
  };

  const loadFutureWork = async () => {
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setFutureLoading(true); setError("");
    try {
      const res = await apiFetch("/api/futurework", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Future Work 提取失败");
      setFutureWork(data.items || []);
      setAiSummary(data.summary || "");
    } catch (e) { setError(e instanceof Error ? e.message : "Future Work 提取失败"); }
    finally { setFutureLoading(false); }
  };

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void loadFutureWork(); void loadGraph(); }, []);

      const option = (graph && graph.nodes && graph.nodes.length > 0) ? {
    backgroundColor: '#0b1120',
    tooltip: {
      backgroundColor: 'rgba(15, 23, 42, 0.9)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc' },
      formatter: (params: any) => {
        const data = params?.data ?? {};

        if (params?.dataType === "node") {
          const name = escapeHtml(data.name);
          const pagerank = Number(data.pagerank ?? 0);
          const inDegree = Number(data.in_degree ?? 0);

          const safePagerank = Number.isFinite(pagerank)
            ? pagerank.toFixed(4)
            : "0.0000";

          const safeInDegree = Number.isFinite(inDegree)
            ? inDegree
            : 0;

          return `<div class="font-bold text-sm mb-1 text-white">${name}</div>
            <div class="text-xs text-slate-300">PageRank: <span class="text-blue-400 font-bold">${safePagerank}</span></div>
            <div class="text-xs text-slate-300">被引频次: <span class="text-green-400 font-bold">${safeInDegree}</span></div>`;
        }

        const source = escapeHtml(data.source);
        const target = escapeHtml(data.target);

        return `<div class="text-xs text-slate-300">${source} → ${target}</div>`;
      }
    },
    animationDuration: 1200,
    animationEasingUpdate: 'quinticInOut',
    series: [{
      type: "graph", layout: "force", roam: true, draggable: true,
      data: graph.nodes.map((node) => {
        const pr = node.pagerank ?? 0;
        let color = "#3b82f6";
        let size = Math.max(30, Math.min(100, node.size * 1.1));
        let isCore = pr >= 0.3;
        let isSecondary = pr >= 0.15;

        if (isCore) { color = "#ef4444"; size = 90; }
        else if (isSecondary) { color = "#f59e0b"; size = 70; }
        else if (pr === 0) { color = "#64748b"; size = 35; }

        return {
          ...node,
          symbolSize: size,
          itemStyle: {
            color,
            borderColor: '#ffffff',
            borderWidth: isCore ? 2 : 1,
            shadowBlur: isCore ? 30 : 10,
            shadowColor: color
          },
          label: {
            show: true,
            position: "bottom",
            distance: 10,
            fontSize: isCore ? 14 : 12,
            fontWeight: isCore ? "bold" : "normal",
            color: "#e2e8f0",
            formatter: (p: any) => p.name.length > 20 ? p.name.substring(0, 20) + "..." : p.name
          },
          labelLayout: { hideOverlap: true },
        };
      }),
      links: graph.edges,
      edgeSymbol: ["none", "arrow"],
      edgeSymbolSize: 8,
      lineStyle: { color: "#475569", opacity: 0.6, width: 1.5, curveness: 0.2 },
      force: { repulsion: 1500, edgeLength: 150, gravity: 0.15 },
      zoom: 1,
      emphasis: {
        focus: "adjacency",
        lineStyle: { width: 3, color: "#93c5fd", opacity: 1 },
        itemStyle: { shadowBlur: 40, shadowColor: '#ffffff' }
      },
    }],
  } : {};

  return (
    <div className="mx-auto min-h-full max-w-7xl space-y-6 bg-slate-50 p-6 md:p-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div><div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-600"><Network className="h-4 w-4" /> Citation Intelligence</div><h1 className="text-3xl font-bold text-slate-900">引用关系与研究脉络</h1><p className="mt-2 text-sm text-slate-500">通过 DOI / 论文标题线索构建本地引用网络，并计算 PageRank 与中介中心性。</p></div>
        <button onClick={() => void loadGraph()} disabled={graphLoading} className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{graphLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Network className="h-4 w-4" />} 生成图谱</button>
      </div>
      {error && <p className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap gap-5 border-b border-slate-100 px-5 py-4 text-sm text-slate-600">
          <span>论文节点：<b className="text-slate-900">{graph?.metrics?.paper_count ?? "—"}</b></span>
          <span>引用边：<b className="text-slate-900">{graph?.metrics?.citation_edges ?? "—"}</b></span>
          <span className="text-xs text-slate-400">红色节点表示相对较高的 PageRank</span>
        </div>
        <div className="h-[520px] p-3">
          {graphLoading ? (
            <div className="relative flex h-full w-full items-center justify-center overflow-hidden rounded-lg bg-[#0b1120]">
              {/* 骨架占位节点 */}
              <div className="absolute h-16 w-16 animate-pulse rounded-full bg-slate-700/50 shadow-[0_0_30px_rgba(59,130,246,0.3)]"></div>
              <div className="absolute left-1/4 top-1/3 h-10 w-10 animate-pulse rounded-full bg-slate-700/50 animation-delay-150"></div>
              <div className="absolute right-1/3 bottom-1/4 h-12 w-12 animate-pulse rounded-full bg-slate-700/50 animation-delay-300"></div>
              <div className="absolute right-1/4 top-1/4 h-8 w-8 animate-pulse rounded-full bg-slate-700/50 animation-delay-500"></div>

              <div className="z-10 flex flex-col items-center gap-3 rounded-2xl bg-slate-900/80 px-6 py-4 backdrop-blur-md border border-slate-700 shadow-2xl">
                <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                <p className="text-sm font-medium text-slate-300">正在计算引用拓扑与 PageRank 中心性...</p>
              </div>
            </div>
          ) : graph ? (
            (graph.nodes && graph.nodes.length > 0) ? (
              <ReactECharts option={option} style={{ width: '100%', height: '100%' }} notMerge lazyUpdate />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-slate-400">
                <Network className="h-10 w-10 text-slate-300" />
                <p>当前文献库尚未识别到可连接的引用关系。</p>
                <p className="text-xs">建议上传同一领域的多篇相关论文后再试。</p>
              </div>
            )
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-slate-400">
              <FileText className="h-10 w-10 text-slate-300" />
              <p>文献库为空，请先上传论文。</p>
            </div>
          )}
        </div>
      </section>
      {graph?.metrics?.top_foundational_papers?.length ? <section className="rounded-2xl border border-slate-200 bg-white p-5"><h2 className="mb-3 font-semibold text-slate-900">核心候选论文（PageRank 排序）</h2><div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{graph.metrics.top_foundational_papers.map((item) => <div key={item.id} className="rounded-xl bg-slate-50 p-4"><div className="font-medium text-slate-800">{item.name}</div><div className="mt-2 text-xs text-slate-500">PageRank {item.pagerank} · 入度 {item.in_degree}</div></div>)}</div></section> : null}
      <section>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-amber-500" /><h2 className="text-xl font-bold text-slate-900">Future Work 与研究局限</h2></div><button onClick={() => void loadFutureWork()} disabled={futureLoading} className="inline-flex items-center gap-2 rounded-lg border border-blue-200 bg-white px-3 py-2 text-xs font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-50">{futureLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} 重新提取</button></div>

        {aiSummary && (
          <div className="mb-6 rounded-2xl border border-indigo-200 bg-indigo-50/50 p-6">
            <h3 className="mb-3 flex items-center gap-2 text-lg font-bold text-indigo-900">
              <Lightbulb className="h-5 w-5 text-indigo-600" />
              AI 综合洞察：未被解决的科研方向
            </h3>
            <div className="whitespace-pre-wrap text-sm leading-7 text-indigo-950">
              {aiSummary.replace(/\*\*/g, '')}
            </div>
          </div>
        )}

        {futureLoading ? (
          <div className="grid gap-4 md:grid-cols-2">
            {[1, 2].map((i) => (
              <div key={i} className="animate-pulse rounded-2xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
                <div className="h-5 w-1/3 rounded bg-slate-200"></div>
                <div className="h-20 w-full rounded bg-slate-100"></div>
                <div className="h-20 w-full rounded bg-slate-100"></div>
              </div>
            ))}
          </div>
        ) : futureWork.length ? (
          <div className="grid gap-4 md:grid-cols-2">
            {futureWork.map((item) => (
              <article key={item.paper_id} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="mb-3 flex items-center gap-2 font-semibold text-slate-800">
                  <FileText className="h-4 w-4 text-blue-600" />
                  {item.filename}
                </h3>
                <ul className="space-y-2">
                  {item.future_work.map((line, i) => (
                    <li key={i} className="rounded-lg border border-amber-100 bg-amber-50/50 p-3 text-sm leading-6 text-slate-700">
                      <div>{line}</div>
                      {item.future_work_evidence?.[i] && (
                        <div className="mt-2 text-[11px] text-amber-700">原文位置：约第 {item.future_work_evidence[i].page} 页</div>
                      )}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400">暂无匹配内容。建议先上传包含 Future Work、Limitations 或 Conclusion 章节的英文论文。</p>
        )}

      </section>
    </div>
  );
}
