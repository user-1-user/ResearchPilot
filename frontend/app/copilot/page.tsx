"use client";




import { useEffect, useState } from "react";
import { FileSpreadsheet, PenTool, Loader2, Sparkles, Download, BookOpen } from "lucide-react";
import { apiFetch } from "@/lib/api";

type PlotResult = { image_url: string; caption: string; rows?: number; x_column?: string; y_column?: string };


const cleanMarkdown = (md: string) => {
  if (!md) return "";
  return md
    .replace(/^### (.*$)/gim, '【$1】')
    .replace(/^## (.*$)/gim, '【$1】')
    .replace(/^# (.*$)/gim, '【$1】')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/^> (.*$)/gim, '💡 $1')
    .replace(/^- (.*$)/gim, '● $1')
    .replace(/^\|?(.*?)\|$/gim, (match) => match.replace(/\|/g, ' ')) // 简单处理表格线
    .replace(/^---$/gim, '————————');
};

export default function CopilotPage() {
  const [activeTab, setActiveTab] = useState<'csv' | 'framework' | 'review' | 'idea'>('csv');
  const [plot, setPlot] = useState<PlotResult | null>(null);
  const [plotImageUrl, setPlotImageUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (plotImageUrl) URL.revokeObjectURL(plotImageUrl);
    };
  }, [plotImageUrl]);
  const [uploading, setUploading] = useState(false);
  const [idea, setIdea] = useState("把大模型应用到医学影像的病灶自动分割中，提升小目标识别精度。");
  const [framework, setFramework] = useState("");
  const [writing, setWriting] = useState(false);
  const [reviewReport, setReviewReport] = useState("");
  const [reviewing, setReviewing] = useState(false);
  const [gapReport, setGapReport] = useState("");
  const [validating, setValidating] = useState(false);
  const [recentWorks, setRecentWorks] = useState<any[]>([]);
  const [error, setError] = useState("");

  const handleCsvUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setUploading(true); setError("");
    try {
      const form = new FormData(); form.append("file", file);
      const res = await apiFetch("/api/csv/plot", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "CSV 解析失败");
      const imageResponse = await apiFetch(data.image_url);
      if (!imageResponse.ok) {
        throw new Error("???????????? CSV");
      }
      const imageBlob = await imageResponse.blob();
      const objectUrl = URL.createObjectURL(imageBlob);
      setPlotImageUrl(objectUrl);
      setPlot(data);
    } catch (e) { setError(e instanceof Error ? e.message : "CSV 生成失败"); }
    finally { setUploading(false); }
  };

  const handleWrite = async () => {
    if (!idea.trim()) { setError("请先输入科研 Idea。"); return; }
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setWriting(true); setError(""); setFramework("");
    try {
      const res = await apiFetch("/api/write/framework", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ idea }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "论文框架生成失败");
      setFramework(data.framework || "");
    } catch (e) { setError(e instanceof Error ? e.message : "生成失败"); }
    finally { setWriting(false); }
  };

  const generateReview = async () => {
    if (!idea.trim()) { setError("请先输入综述主题。 "); return; }
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setReviewing(true); setReviewReport(""); setError("");
    try {
      const res = await apiFetch("/api/research/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic: idea }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "文献综述生成失败");
      setReviewReport(data.report || "");
    } catch (e) { setError(e instanceof Error ? e.message : "文献综述生成失败"); }
    finally { setReviewing(false); }
  };

  const validateIdea = async () => {
    if (!idea.trim()) { setError("请先输入科研 Idea。"); return; }
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setValidating(true); setGapReport(""); setRecentWorks([]); setError("");
    try {
      const res = await apiFetch("/api/research/validate-idea", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ idea }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "选题验证失败");
      setGapReport(data.analysis || "");
      setRecentWorks(data.recent_arxiv || []);
    } catch (e) { setError(e instanceof Error ? e.message : "选题验证失败"); }
    finally { setValidating(false); }
  };

  return (
    <div className="mx-auto flex h-full max-w-7xl flex-col bg-slate-50 p-6 md:p-8">
      <div className="mb-6 shrink-0">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600"><Sparkles className="h-4 w-4" /> Academic Copilot</div>
        <h1 className="text-3xl font-bold text-slate-900">科研写作与数据工作台</h1>
        <p className="mt-2 text-sm text-slate-500">使用可追溯的文献证据辅助选题、写作和实验结果表达。</p>
      </div>

      {error && <p className="mb-5 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}

      <div className="flex flex-1 gap-6 overflow-hidden">
        {/* 左侧导航栏 */}
        <aside className="w-56 shrink-0 flex flex-col gap-2">
          <button onClick={() => setActiveTab('csv')} className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-semibold transition ${activeTab === 'csv' ? 'bg-blue-600 text-white shadow-md' : 'bg-white text-slate-600 hover:bg-slate-100 border border-transparent'}`}>
            <FileSpreadsheet className="h-4 w-4" /> 数据可视化
          </button>
          <button onClick={() => setActiveTab('framework')} className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-semibold transition ${activeTab === 'framework' ? 'bg-blue-600 text-white shadow-md' : 'bg-white text-slate-600 hover:bg-slate-100 border border-transparent'}`}>
            <PenTool className="h-4 w-4" /> 论文框架
          </button>
          <button onClick={() => setActiveTab('review')} className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-semibold transition ${activeTab === 'review' ? 'bg-blue-600 text-white shadow-md' : 'bg-white text-slate-600 hover:bg-slate-100 border border-transparent'}`}>
            <BookOpen className="h-4 w-4" /> 主题综述
          </button>
          <button onClick={() => setActiveTab('idea')} className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-semibold transition ${activeTab === 'idea' ? 'bg-blue-600 text-white shadow-md' : 'bg-white text-slate-600 hover:bg-slate-100 border border-transparent'}`}>
            <Sparkles className="h-4 w-4" /> Idea 验证
          </button>
        </aside>

        {/* 右侧主工作区 */}
        <div className="flex-1 overflow-y-auto rounded-2xl border border-slate-200 bg-white shadow-sm">

          {activeTab === 'csv' && (
            <div className="flex h-full flex-col p-6">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-bold text-slate-800">实验数据可视化</h2>
                  <p className="text-xs text-slate-500 mt-1">上传 CSV，系统自动识别数值列、生成图表并给出学术图注。</p>
                </div>
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700 hover:bg-blue-100 transition">
                  {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileSpreadsheet className="h-4 w-4" />}
                  {uploading ? "正在生成…" : "选择 CSV 文件"}
                  <input type="file" accept=".csv,text/csv" disabled={uploading} onChange={handleCsvUpload} className="hidden" />
                </label>
              </div>
              <div className="flex flex-1 flex-col">
                {plot ? (
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <img src={plotImageUrl || undefined} alt="Generated chart" className="mx-auto max-h-80 rounded-lg border border-slate-200 bg-white" />
                    <div className="mt-4 rounded-lg bg-white p-4 border border-slate-100">
                      <div className="mb-2 text-xs font-semibold text-slate-700">Figure Caption & Insight</div>
                      <p className="whitespace-pre-wrap text-sm leading-6 text-slate-600">{cleanMarkdown(plot.caption)}</p>
                      <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
                        <span>{plot.rows} rows · {plot.x_column} vs {plot.y_column}</span>
                        <a href={plotImageUrl || undefined} download className="inline-flex items-center gap-1 text-blue-600 hover:underline"><Download className="h-3.5 w-3.5" /> 下载图表</a>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-1 items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 text-sm text-slate-400">
                    暂无图表，请上传实验 CSV 数据。
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'framework' && (
            <div className="flex h-full flex-col p-6">
              <div className="mb-4">
                <h2 className="text-lg font-bold text-slate-800">有证据约束的论文框架生成</h2>
                <p className="text-xs text-slate-500 mt-1">系统先从文献库检索相关原文，再生成框架。没有证据的引用会被标记为待补充。</p>
              </div>
              <textarea value={idea} onChange={(e) => setIdea(e.target.value)} rows={3} className="mb-3 resize-none rounded-xl border border-slate-200 p-3 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100" placeholder="描述你的研究问题、方法、数据集和目标…" />
              <button onClick={() => void handleWrite()} disabled={writing || !idea.trim()} className="mb-4 inline-flex w-fit items-center gap-2 rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">
                {writing ? <Loader2 className="h-4 w-4 animate-spin" /> : <PenTool className="h-4 w-4" />}{writing ? "正在检索并生成…" : "生成论文框架"}
              </button>
              <div className="flex-1 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-5">
                {framework ? (
                  <div className="whitespace-pre-wrap font-sans text-sm leading-7 text-slate-700">
                    {cleanMarkdown(framework)}
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-400">生成的框架将在这里显示。</div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'review' && (
            <div className="flex h-full flex-col p-6">
              <div className="mb-4">
                <h2 className="text-lg font-bold text-slate-800">多文献主题综述生成</h2>
                <p className="text-xs text-slate-500 mt-1">基于文献库当前可检索到的证据生成主题综述，事实性表述要求附证据编号。</p>
              </div>
              <textarea value={idea} onChange={(e) => setIdea(e.target.value)} rows={3} className="mb-3 resize-none rounded-xl border border-slate-200 p-3 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100" placeholder="输入你想综述的研究主题（复用上方输入框逻辑，请先输入主题）…" />
              <button onClick={() => void generateReview()} disabled={reviewing || !idea.trim()} className="mb-4 inline-flex w-fit items-center gap-2 rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50">
                {reviewing ? <Loader2 className="h-4 w-4 animate-spin" /> : <BookOpen className="h-4 w-4" />}{reviewing ? "正在检索并归纳…" : "生成主题综述"}
              </button>
              <div className="flex-1 overflow-auto whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 p-5 text-sm leading-7 text-slate-700">
                {reviewReport ? (
                  <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
                    {cleanMarkdown(reviewReport)}
                  </div>
                ) : (
                  "综述报告将在此处显示。覆盖范围受当前文献库内容与检索结果限制。"
                )}
              </div>
            </div>
          )}

          {activeTab === 'idea' && (
            <div className="flex h-full flex-col p-6">
              <div className="mb-4">
                <h2 className="text-lg font-bold text-slate-800">科研选题验证与研究空白探索</h2>
                <p className="text-xs text-slate-500 mt-1">结合本地论文证据与近期 arXiv 结果，识别相近工作、潜在研究空白和创新风险。</p>
              </div>
              <textarea value={idea} onChange={(e) => setIdea(e.target.value)} rows={3} className="mb-3 resize-none rounded-xl border border-slate-200 p-3 text-sm outline-none focus:border-violet-400 focus:ring-2 focus:ring-violet-100" placeholder="详细描述你的科研 Idea…" />
              <button onClick={() => void validateIdea()} disabled={validating || !idea.trim()} className="mb-4 inline-flex w-fit items-center gap-2 rounded-xl bg-violet-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:opacity-50">
                {validating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}{validating ? "正在对照近期研究…" : "验证当前 Idea 的研究空白"}
              </button>
              <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden xl:grid-cols-[minmax(0,1fr)_320px]">
                <div className="overflow-auto whitespace-pre-wrap rounded-xl border border-violet-100 bg-violet-50 p-5 text-sm leading-7 text-slate-800">
                  {gapReport ? (
                  <div className="whitespace-pre-wrap text-sm leading-7 text-slate-800">
                    {cleanMarkdown(gapReport)}
                  </div>
                ) : (
                  "验证结果将在此处显示。"
                )}
                </div>
                <aside className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <h3 className="mb-3 shrink-0 text-sm font-semibold text-slate-800">近期相关 arXiv 工作</h3>
                  <div className="flex-1 overflow-auto space-y-3">
                    {recentWorks.length ? recentWorks.slice(0, 5).map((work) => (
                      <div key={work.id} className="border-b border-slate-200 pb-3 last:border-0">
                        <a href={work.url} target="_blank" rel="noreferrer" className="text-sm font-medium text-blue-700 hover:underline">{work.title}</a>
                        <p className="mt-1 text-xs text-slate-400">{work.id} · {work.published?.slice(0, 10)}</p>
                      </div>
                    )) : <p className="text-xs leading-5 text-slate-400">提交验证后，相关论文将显示在这里。</p>}
                  </div>
                </aside>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
