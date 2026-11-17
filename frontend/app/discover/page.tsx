"use client";

import { useEffect, useState } from "react";
import { Compass, Download, Loader2, Search, ExternalLink, Sparkles, ThumbsUp, ThumbsDown } from "lucide-react";
import { apiFetch } from "@/lib/api";

type ArxivPaper = { id: string; title: string; summary: string; published: string; authors: string[]; url: string; pdf_url: string };

export default function DiscoverPage() {
  const [query, setQuery] = useState("large language model agent scientific research");
  const [items, setItems] = useState<ArxivPaper[]>([]);
  const [recommendationReason, setRecommendationReason] = useState("");
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [notice, setNotice] = useState("");

  const search = async (q = query) => {
    if (!q.trim()) return;
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setLoading(true); setNotice("");
    try {
      const res = await apiFetch(`/api/arxiv/search?q=${encodeURIComponent(q)}&limit=10`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "arXiv 检索失败");
      setItems(data.items || []);
    } catch (e) { setNotice(e instanceof Error ? e.message : "检索失败"); }
    finally { setLoading(false); }
  };

  const loadRecommendations = async () => {
    if (!localStorage.getItem("yantu_token")) return;
    setLoading(true);
    try {
      const res = await apiFetch("/api/recommendations?limit=8");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "推荐加载失败");
      setItems(data.items || []); setRecommendationReason(data.reason || "");
    } catch (e) { setNotice(e instanceof Error ? e.message : "推荐加载失败"); }
    finally { setLoading(false); }
  };

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void loadRecommendations(); }, []);

  const downloadPaper = async (paper: ArxivPaper) => {
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setDownloading(paper.id); setNotice("");
    try {
      const res = await apiFetch("/api/arxiv/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ arxiv_id: paper.id }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "下载入库失败");
      setNotice(data.duplicates?.length ? `${paper.id} 已下载，系统检测到重复版本，请到文献库处理。` : `${paper.id} 已下载并完成解析索引。`);
    } catch (e) { setNotice(e instanceof Error ? e.message : "下载失败"); }
    finally { setDownloading(null); }
  };

  return (
    <div className="mx-auto min-h-full max-w-6xl space-y-6 bg-slate-50 p-6 md:p-8">
      <div><div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600"><Compass className="h-4 w-4" /> Literature Discovery</div><h1 className="text-3xl font-bold text-slate-900">论文发现与智能推荐</h1><p className="mt-2 text-sm text-slate-500">检索 arXiv 最新工作，查看摘要与来源，并一键下载到个人文献库。</p></div>
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-3 md:flex-row"><input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void search()} className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none focus:border-blue-400" placeholder="输入研究方向或关键词…" /><button onClick={() => void search()} disabled={loading} className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} 检索论文</button><button onClick={() => void loadRecommendations()} disabled={loading} className="rounded-xl border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50">按文献库推荐</button></div>
        {recommendationReason && <p className="mt-3 text-xs text-slate-400">{recommendationReason}</p>}
      </section>
      {notice && (
        <div className={`mt-4 flex items-start gap-3 rounded-xl border p-4 text-sm ${
          notice.includes("失败") || notice.includes("Error")
            ? "border-rose-200 bg-rose-50 text-rose-800"
            : "border-emerald-200 bg-emerald-50 text-emerald-800"
        }`}>
          <div className="mt-0.5">
            {notice.includes("失败") || notice.includes("Error") ? "❌" : "✅"}
          </div>
          <div>
            <p className="font-semibold">{notice.includes("失败") || notice.includes("Error") ? "操作受限/失败" : "执行成功"}</p>
            <p className="mt-1 opacity-80">{notice}</p>
          </div>
        </div>
      )}
      <div className="flex items-center justify-between"><h2 className="font-semibold text-slate-900">检索结果</h2><span className="text-xs text-slate-400">{items.length} 篇</span></div>
      <div className="space-y-4">
        {items.map((paper) => <article key={paper.id} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1"><h3 className="text-base font-semibold leading-6 text-slate-900">{paper.title}</h3><p className="mt-1 text-xs text-slate-400">{paper.id} · {paper.published?.slice(0, 10)} · {paper.authors?.slice(0, 5).join(", ")}</p></div>
            <div className="flex shrink-0 gap-2"><a href={paper.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50"><ExternalLink className="h-3.5 w-3.5" /> arXiv</a><button onClick={() => void downloadPaper(paper)} disabled={downloading === paper.id} className="inline-flex items-center gap-1 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{downloading === paper.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} 下载入库</button></div>
          </div>
          <p className="mt-3 line-clamp-5 text-sm leading-6 text-slate-600">{paper.summary}</p>
        </article>)}
        {!items.length && !loading && <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white py-16 text-center text-sm text-slate-400"><Sparkles className="mx-auto mb-3 h-7 w-7 text-slate-300" />暂时没有论文结果，请搜索研究主题。</div>}
      </div>
    </div>
  );
}
