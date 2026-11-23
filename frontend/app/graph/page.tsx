"use client";

import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  ArrowDownRight,
  ArrowUpRight,
  BookOpen,
  FileText,
  Lightbulb,
  Loader2,
  Network,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { apiFetch } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type GraphNode = {
  id: string;
  name: string;
  filename?: string;
  size: number;
  pagerank?: number;
  betweenness?: number;
  in_degree?: number;
  out_degree?: number;
};

type GraphEdge = {
  source: string;
  target: string;
  relation?: string;
  evidence?: string;
};

type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics?: {
    paper_count?: number;
    citation_edges?: number;
    top_foundational_papers?: GraphNode[];
    isolated_papers?: string[];
  };
};

type FutureItem = {
  paper_id: string;
  filename: string;
  future_work: string[];
  future_work_evidence?: {
    page: number | null;
    quote: string;
  }[];
};

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

function evidenceLabel(value?: string) {
  if (value === "reference DOI") return "参考文献 DOI 匹配";
  if (value === "reference title") return "参考文献标题匹配";
  return value || "引用关系";
}

export default function GraphPage() {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState("");
  const [futureWork, setFutureWork] = useState<FutureItem[]>([]);
  const [aiSummary, setAiSummary] = useState("");
  const [futureLoading, setFutureLoading] = useState(false);
  const [futureError, setFutureError] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const normalizedAiSummary = aiSummary
    .replace(/\\-/g, "-")
    .replace(/\s+-\s*(?=(?:方向|研究方向)\s*[一二三四五六七八九十\d])/g, "\n- ");

  const selectedNode = graph?.nodes.find(
    (node) => node.id === selectedNodeId
  );

  const relatedEdges =
    graph?.edges.filter(
      (edge) =>
        edge.source === selectedNodeId ||
        edge.target === selectedNodeId
    ) ?? [];

  const ensureAuth = () => {
    if (localStorage.getItem("yantu_token")) return true;
    window.dispatchEvent(new Event("open-login-modal"));
    return false;
  };

  const loadGraph = async () => {
    if (!ensureAuth()) return;

    setGraphLoading(true);
    setGraphError("");

    try {
      const res = await apiFetch("/api/graph/build", {
        method: "POST",
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "图谱生成失败");
      }

      setGraph(data);
      setSelectedNodeId(null);
    } catch (error) {
      setGraphError(
        error instanceof Error ? error.message : "图谱生成失败"
      );
    } finally {
      setGraphLoading(false);
    }
  };

  const loadFutureWork = async () => {
    if (!ensureAuth()) return;

    setFutureLoading(true);
    setFutureError("");

    try {
      const res = await apiFetch("/api/futurework", {
        method: "POST",
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "研究方向提取失败");
      }

      setFutureWork(data.items || []);
      setAiSummary(data.summary || "");
    } catch (error) {
      setFutureError(
        error instanceof Error ? error.message : "研究方向提取失败"
      );
    } finally {
      setFutureLoading(false);
    }
  };

  useEffect(() => {
    if (!localStorage.getItem("yantu_token")) {
      window.dispatchEvent(new Event("open-login-modal"));
      return;
    }

    void loadGraph();
    void loadFutureWork();
  }, []);

  const chartEvents = {
    click: (params: any) => {
      if (params?.dataType === "node" && params?.data?.id) {
        setSelectedNodeId(String(params.data.id));
      }
    },
  };

  const nodes = graph?.nodes ?? [];
  const rankedNodes = [...nodes].sort(
    (a, b) => (b.pagerank ?? 0) - (a.pagerank ?? 0)
  );

  const coreCount = Math.max(1, Math.ceil(nodes.length * 0.2));
  const secondaryCount = Math.max(
    coreCount,
    Math.ceil(nodes.length * 0.5)
  );

  const rankMap = new Map(
    rankedNodes.map((node, index) => [node.id, index])
  );

  const option =
    graph && nodes.length > 0
      ? {
          backgroundColor: "#ffffff",
          animationDuration: 700,
          animationEasingUpdate: "cubicOut",
          tooltip: { show: false,
            trigger: "item",
            backgroundColor: "#ffffff",
            borderColor: "#e2e8f0",
            borderWidth: 1,
            padding: 12,
            textStyle: {
              color: "#334155",
              fontSize: 12,
            },
            formatter: (params: any) => {
              const data = params?.data ?? {};

              if (params?.dataType === "node") {
                return `
                  <div style="max-width:260px">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:8px">
                      ${escapeHtml(data.name)}
                    </div>
                    <div>网络影响力：${Number(data.pagerank ?? 0).toFixed(6)}</div>
                    <div>被引用次数：${Number(data.in_degree ?? 0)}</div>
                    <div>引用库内论文数：${Number(data.out_degree ?? 0)}</div>
                    <div style="margin-top:6px;color:#64748b">
                      点击节点查看详细关系
                    </div>
                  </div>
                `;
              }

              return `
                <div style="font-size:12px">
                  ${escapeHtml(data.source)} → ${escapeHtml(data.target)}
                  <div style="margin-top:4px;color:#64748b">
                    引用关系
                  </div>
                </div>
              `;
            },
          },
          series: [
            {
              type: "graph",
              layout: "force",
              roam: true,
              draggable: true,
              data: nodes.map((node) => {
                const rank = rankMap.get(node.id) ?? nodes.length;
                const pagerank = node.pagerank ?? 0;
                const hasEdges = (graph.edges?.length ?? 0) > 0;

                let color = "#94a3b8";
                if (hasEdges && rank < coreCount) {
                  color = "#d97706";
                } else if (hasEdges && rank < secondaryCount) {
                  color = "#3b82f6";
                }

                const size = Math.max(
                  22,
                  Math.min(54, 22 + Math.sqrt(Math.max(0, pagerank)) * 180)
                );

                const isSelected = selectedNodeId === node.id;

                return {
                  ...node,
                  symbolSize: size,
                  itemStyle: {
                    color,
                    borderColor: isSelected ? "#0f172a" : "#ffffff",
                    borderWidth: isSelected ? 4 : 2,
                    shadowBlur: isSelected ? 16 : 5,
                    shadowColor: color,
                  },
                  label: {
                    show: rank < Math.min(12, nodes.length),
                    position: "bottom",
                    distance: 8,
                    fontSize: rank < coreCount ? 12 : 10,
                    fontWeight: rank < coreCount ? "bold" : "normal",
                    color: "#475569",
                    formatter: (p: any) =>
                      String(p.name ?? "").length > 18
                        ? String(p.name).substring(0, 18) + "…"
                        : p.name,
                  },
                  labelLayout: {
                    hideOverlap: true,
                  },
                };
              }),
              links: graph.edges.map((edge) => ({
                ...edge,
              })),
              edgeSymbol: ["none", "arrow"],
              edgeSymbolSize: 9,
              lineStyle: {
                color: "#94a3b8",
                opacity: 0.65,
                width: 1.4,
                curveness: 0.16,
              },
              force: {
                repulsion: 950,
                edgeLength: 125,
                gravity: 0.12,
                friction: 0.6,
              },
              emphasis: {
                focus: "adjacency",
                lineStyle: {
                  width: 2.8,
                  color: "#2563eb",
                  opacity: 1,
                },
                itemStyle: {
                  shadowBlur: 18,
                  shadowColor: "#60a5fa",
                },
              },
            },
          ],
        }
      : {};

  return (
    <div className="mx-auto min-h-full max-w-[1500px] space-y-7 bg-slate-50 px-5 py-7 md:px-8 md:py-9">
      {/* 页面标题 */}
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wide text-blue-700">
            <Network className="h-4 w-4" />
            RESEARCH KNOWLEDGE GRAPH
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
            引用关系图谱
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            从已上传论文的参考文献中识别引用关系，帮助你发现关键论文与研究联系。
          </p>
        </div>

        <button
          onClick={() => void loadGraph()}
          disabled={graphLoading}
          className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {graphLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
          {graphLoading ? "正在生成…" : "重新生成图谱"}
        </button>
      </section>

      {/* 图谱统计 */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <p className="text-xs text-slate-500">论文数量</p>
          <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">
            {graph?.metrics?.paper_count ?? "—"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <p className="text-xs text-slate-500">识别出的引用关系</p>
          <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">
            {graph?.metrics?.citation_edges ?? "—"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <p className="text-xs text-slate-500">存在引用联系的论文</p>
          <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">
            {graph
              ? nodes.filter(
                  (node) =>
                    (node.in_degree ?? 0) + (node.out_degree ?? 0) > 0
                ).length
              : "—"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <p className="text-xs text-slate-500">暂无库内引用联系</p>
          <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">
            {graph?.metrics?.isolated_papers?.length ?? "—"}
          </p>
        </div>
      </section>

      {/* 引用网络 */}
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 px-5 py-4">
          <div>
            <h2 className="font-semibold text-slate-900">论文引用网络</h2>
            <p className="mt-1 text-xs text-slate-500">
              点击论文节点查看详情，拖动节点调整布局，滚轮缩放图谱。
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-600">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-amber-600" />
              网络影响力较高
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-blue-500" />
              网络影响力居中
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-slate-400" />
              其他论文
            </span>
          </div>
        </div>

        <div className="border-b border-slate-100 bg-slate-50/70 px-5 py-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-500">
            <span>
              <b className="text-slate-700">节点大小：</b>
              与网络影响力相关，越大表示相对值越高
            </span>
            <span>
              <b className="text-slate-700">箭头方向：</b>
              引用论文 → 被引用论文
            </span>
          </div>
        </div>

        {graphError && (
          <div className="m-5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {graphError}
          </div>
        )}

        <div className="h-[460px] p-2 md:h-[580px] md:p-4">
          {graphLoading ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 rounded-xl bg-slate-50">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              <p className="text-sm font-medium text-slate-700">
                正在分析论文引用关系…
              </p>
              <p className="text-xs text-slate-400">
                请稍候，处理时间取决于文献数量
              </p>
            </div>
          ) : graph && nodes.length > 0 ? (
            <ReactECharts
              option={option}
              onEvents={chartEvents}
              style={{ width: "100%", height: "100%" }}
              notMerge
              lazyUpdate
            />
          ) : graph ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 rounded-xl bg-slate-50 px-5 text-center">
              <Network className="h-10 w-10 text-slate-300" />
              <p className="font-medium text-slate-700">
                暂未发现可连接的引用关系
              </p>
              <p className="max-w-md text-sm leading-6 text-slate-500">
                当前文献库中有论文，但没有识别到它们之间的引用联系。上传同一研究领域的多篇论文后，可以再次生成图谱。
              </p>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-3 rounded-xl bg-slate-50">
              <BookOpen className="h-10 w-10 text-slate-300" />
              <p className="text-sm text-slate-500">
                正在等待图谱数据，或请先上传论文
              </p>
            </div>
          )}
        </div>
      </section>

      {/* 选中论文详情 */}
      {selectedNode && (
        <section className="overflow-hidden rounded-2xl border border-blue-200 bg-white shadow-sm">
          <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
            <div className="min-w-0">
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-blue-700">
                <FileText className="h-4 w-4" />
                论文详情
              </div>
              <h2 className="break-words text-lg font-bold leading-7 text-slate-900">
                {selectedNode.name}
              </h2>
              <p className="mt-1 break-all text-xs text-slate-400">
                {selectedNode.filename || selectedNode.id}
              </p>
            </div>
            <button
              onClick={() => setSelectedNodeId(null)}
              aria-label="关闭论文详情"
              className="shrink-0 rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3 p-5 md:grid-cols-4">
            <div className="rounded-xl bg-blue-50 p-3">
              <p className="text-xs text-blue-700">网络影响力</p>
              <p className="mt-2 text-lg font-bold tabular-nums text-slate-900">
                {(selectedNode.pagerank ?? 0).toFixed(6)}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                在当前引用网络中的相对重要性
              </p>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <p className="text-xs text-slate-500">被引用次数</p>
              <p className="mt-2 text-lg font-bold tabular-nums text-slate-900">
                {selectedNode.in_degree ?? 0}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                文献库内有多少论文引用它
              </p>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <p className="text-xs text-slate-500">引用库内论文数</p>
              <p className="mt-2 text-lg font-bold tabular-nums text-slate-900">
                {selectedNode.out_degree ?? 0}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                它引用了多少篇库内论文
              </p>
            </div>
            <div className="rounded-xl bg-slate-50 p-3">
              <p className="text-xs text-slate-500">研究桥梁作用</p>
              <p className="mt-2 text-lg font-bold tabular-nums text-slate-900">
                {(selectedNode.betweenness ?? 0).toFixed(4)}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                连接不同论文群体的程度
              </p>
            </div>
          </div>

          <div className="border-t border-slate-100 px-5 py-5">
            <h3 className="mb-3 text-sm font-semibold text-slate-800">
              与这篇论文相关的引用关系
            </h3>

            {relatedEdges.length > 0 ? (
              <div className="space-y-2">
                {relatedEdges.map((edge, index) => {
                  const outgoing = edge.source === selectedNode.id;
                  const otherId = outgoing ? edge.target : edge.source;
                  const otherNode = graph?.nodes.find(
                    (node) => node.id === otherId
                  );

                  return (
                    <div
                      key={`${edge.source}-${edge.target}-${index}`}
                      className="flex flex-col gap-2 rounded-xl border border-slate-100 p-3 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="flex min-w-0 items-center gap-2">
                        {outgoing ? (
                          <ArrowUpRight className="h-4 w-4 shrink-0 text-blue-600" />
                        ) : (
                          <ArrowDownRight className="h-4 w-4 shrink-0 text-emerald-600" />
                        )}
                        <span className="shrink-0 rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600">
                          {outgoing ? "引用了" : "被引用"}
                        </span>
                        <span className="break-words text-sm font-medium text-slate-800">
                          {otherNode?.name || otherId}
                        </span>
                      </div>
                      <span className="text-xs text-slate-400">
                        {evidenceLabel(edge.evidence || edge.relation)}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="rounded-xl bg-slate-50 px-4 py-5 text-sm text-slate-500">
                这篇论文目前没有识别到其他库内论文与它的直接引用关系。
              </p>
            )}
          </div>
        </section>
      )}

      {/* 网络关键论文 */}
      {rankedNodes.length > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="font-semibold text-slate-900">
              网络关键论文
            </h2>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              按 PageRank 从高到低排列，反映论文在当前文献引用网络中的相对影响力，不代表论文质量或学术权威性。
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {rankedNodes.slice(0, 5).map((item, index) => (
              <button
                key={item.id}
                onClick={() => setSelectedNodeId(item.id)}
                className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-left transition hover:border-blue-200 hover:bg-blue-50/50"
              >
                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-500">
                    TOP {index + 1}
                  </span>
                  <span className="text-xs tabular-nums text-slate-400">
                    PageRank {(item.pagerank ?? 0).toFixed(4)}
                  </span>
                </div>
                <p className="line-clamp-2 text-sm font-semibold leading-6 text-slate-800">
                  {item.name}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  被引用 {item.in_degree ?? 0} 次 · 引用库内论文{" "}
                  {item.out_degree ?? 0} 篇
                </p>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* 研究方向与局限 */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5 text-amber-500" />
              <h2 className="text-xl font-bold text-slate-900">
                研究方向与局限
              </h2>
            </div>
            <p className="mt-1 text-sm text-slate-500">
              从已上传论文中提取作者提出的未来研究方向和局限。
            </p>
          </div>

          <button
            onClick={() => void loadFutureWork()}
            disabled={futureLoading}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
          >
            {futureLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            重新提取
          </button>
        </div>

        {futureError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {futureError}
          </div>
        )}

        {aiSummary && (
          <div className="rounded-2xl border border-indigo-100 bg-indigo-50/60 p-5 md:p-6">
            <h3 className="mb-3 flex items-center gap-2 font-semibold text-indigo-950">
              <Sparkles className="h-5 w-5 text-indigo-600" />
              AI 综合研究洞察
            </h3>
            <div className="text-sm leading-7 text-indigo-950">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  ul: ({ children }) => (
                    <ul className="my-3 list-disc space-y-3 pl-6">{children}</ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="my-3 list-decimal space-y-3 pl-6">{children}</ol>
                  ),
                  li: ({ children }) => (
                    <li className="pl-1">{children}</li>
                  ),
                  p: ({ children }) => (
                    <p className="my-2">{children}</p>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-semibold text-indigo-950">{children}</strong>
                  ),
                }}
              >
                {normalizedAiSummary}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {futureLoading ? (
          <div className="grid gap-4 md:grid-cols-2">
            {[1, 2].map((item) => (
              <div
                key={item}
                className="animate-pulse rounded-2xl border border-slate-200 bg-white p-5"
              >
                <div className="mb-4 h-4 w-1/3 rounded bg-slate-200" />
                <div className="mb-2 h-16 rounded bg-slate-100" />
                <div className="h-16 rounded bg-slate-100" />
              </div>
            ))}
          </div>
        ) : futureWork.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {futureWork.map((item) => (
              <article
                key={item.paper_id}
                className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
              >
                <h3 className="mb-4 flex items-start gap-2 break-words text-sm font-semibold leading-6 text-slate-800">
                  <FileText className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                  {item.filename}
                </h3>

                <ul className="space-y-3">
                  {item.future_work.map((line, index) => {
                    const evidence = item.future_work_evidence?.[index];

                    return (
                      <li
                        key={index}
                        className="rounded-xl border border-amber-100 bg-amber-50/50 p-3"
                      >
                        <p className="text-sm leading-6 text-slate-700">
                          {line}
                        </p>
                        {evidence && (
                          <p className="mt-2 text-xs text-amber-800">
                            {typeof evidence.page === "number"
                              ? `原文位置：第 ${evidence.page} 页`
                              : "原文位置：未能精确定位，请在原文中检索"}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </article>
            ))}
          </div>
        ) : (
          !futureLoading && (
            <div className="rounded-2xl border border-slate-200 bg-white px-5 py-8 text-center">
              <Lightbulb className="mx-auto mb-3 h-8 w-8 text-slate-300" />
              <p className="text-sm font-medium text-slate-700">
                暂无研究方向提取结果
              </p>
              <p className="mt-1 text-xs text-slate-400">
                上传包含 Future Work、Limitations 或 Conclusion 内容的论文后，可以重新提取。
              </p>
            </div>
          )
        )}
      </section>
    </div>
  );
}
