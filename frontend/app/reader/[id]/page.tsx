"use client";




import { use, useEffect, useRef, useState } from "react";
import { BookOpen, Languages, Loader2, Send, Bot, ChevronLeft, ChevronRight, CheckCircle2, FileText } from "lucide-react";
import { apiFetch, consumeSSE } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type Page = { page: number; text: string; method?: string; quality?: number };
type Paper = { id: string; filename: string; page_count: number; pages: Page[]; metadata?: { title?: string; abstract?: string; sections?: { title: string; page: number; level: number }[] } };



function renderCitations(
  text: string,
  onReferenceClick: (num: string) => void
) {
  if (!text) return null;

  const codePattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;
  const segments = text.split(codePattern);

  return segments.map((segment, segmentIndex) => {
    if (/^(?:```|~~~|`)/.test(segment)) {
      return <span key={`code-${segmentIndex}`}>{segment}</span>;
    }

    const parts = segment.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);

    return (
      <span key={`segment-${segmentIndex}`}>
        {parts.map((part, index) => {
          const match = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);

          if (!match) {
            return <span key={index}>{part}</span>;
          }

          const tokens = match[1].split(/(\s*,\s*)/);

          return (
            <span key={index}>
              [
              {tokens.map((token, tokenIndex) => {
                if (!/^\d+$/.test(token)) {
                  return <span key={tokenIndex}>{token}</span>;
                }

                return (
                  <span
                    key={tokenIndex}
                    role="button"
                    tabIndex={0}
                    onClick={() => onReferenceClick(token)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onReferenceClick(token);
                      }
                    }}
                    className="cursor-pointer rounded px-1 font-medium text-blue-600 hover:bg-blue-50 hover:underline"
                    title={`Jump to reference [${token}]`}
                  >
                    {token}
                  </span>
                );
              })}
              ]
            </span>
          );
        })}
      </span>
    );
  });
}

function renderReferences(
  text: string,
  side: "left" | "right" = "left",
  pageNumber = 0,
  highlightedReference: string | null = null
) {
  if (!text) return null;

  const lines = text.split(/(?<=\n)/);
  const entries: { id: string; content: string }[] = [];
  let preamble = "";
  let current: { id: string; content: string } | null = null;

  // ????????????????????
  for (const line of lines) {
    const match = line.match(/^\s*(?:\*\*)?\[(\d+)\](?:\*\*)?\s*/);

    if (match) {
      if (current) entries.push(current);

      current = {
        id: match[1],
        content: line
      };
    } else if (current) {
      current.content += line;
    } else {
      preamble += line;
    }
  }

  if (current) entries.push(current);

  return (
    <div>
      {preamble ? (
        <div className="whitespace-pre-wrap">{preamble}</div>
      ) : null}

      {entries.map((entry, index) => {
        const targetId = `${side}-${pageNumber}-${entry.id}`;
        const active = highlightedReference === targetId;

        return (
          <div
            key={`${targetId}-${index}`}
            id={`${side}-ref-${pageNumber}-${entry.id}`}
            className={`mb-3 scroll-mt-20 rounded border-l-2 pl-3 transition-colors ${
              active
                ? "border-amber-500 bg-amber-100 ring-1 ring-amber-300"
                : "border-slate-200 hover:border-slate-400"
            }`}
          >
            <div className="whitespace-pre-wrap">
              <strong className="text-slate-800">[{entry.id}]</strong>
              {entry.content.replace(/^\s*(?:\*\*)?\[\d+\](?:\*\*)?\s*/, " ")}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function renderTranslatedContent(
  text: string,
  onReferenceClick: (num: string) => void,
  pageNumber: number,
  isReferencePage: boolean,
  highlightedReference: string | null
) {
  if (!text) return null;

  const heading = text.match(
    /(?:^|\n)\s*(\u53c2\u8003\u6587\u732e|\u53c2\u8003\u8d44\u6599|References|Bibliography)\s*[:\uFF1A]?\s*(?:\n|$)/im
  );

  if (!heading || heading.index === undefined) {
    if (isReferencePage) {
      return renderReferences(
        text,
        "right",
        pageNumber,
        highlightedReference
      );
    }

    return renderCitations(text, onReferenceClick);
  }

  const bodyStart = heading.index + heading[0].length;

  return (
    <>
      {text.slice(0, heading.index)
        ? renderCitations(text.slice(0, heading.index), onReferenceClick)
        : null}

      <div className="mt-4 border-t border-slate-200 pt-3">
        <h3 className="mb-3 text-lg font-bold">{heading[1]}</h3>
        {renderReferences(
          text.slice(bodyStart),
          "right",
          pageNumber,
          highlightedReference
        )}
      </div>
    </>
  );
}

export default function ReaderPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const pageSize = 5;
  const [startPage, setStartPage] = useState(1);
  const [pageInput, setPageInput] = useState("");
  const [pendingJumpPage, setPendingJumpPage] = useState<number | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [loading, setLoading] = useState(true);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [showPdf, setShowPdf] = useState(false);
  const [translations, setTranslations] = useState<Record<number, string>>({});
  const [translating, setTranslating] = useState<number | null>(null);
  const [isBatchTranslating, setIsBatchTranslating] = useState(false);
  const [question, setQuestion] = useState("这篇论文的核心贡献是什么？");
  const [answer, setAnswer] = useState("");
  const [steps, setSteps] = useState<any[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const syncTarget = useRef<HTMLElement | null>(null);
  const syncExpectedTop = useRef(0);
  const syncResetTimer = useRef<number | null>(null);
  const syncPauseUntil = useRef(0);
  const [pendingReference, setPendingReference] = useState<number | null>(null);
  const [pendingReferenceSide, setPendingReferenceSide] = useState<'left' | 'right'>('left');
  const [referenceStartPage, setReferenceStartPage] = useState<number | null>(null);
  const [pendingReferencePage, setPendingReferencePage] = useState<number | null>(null);
  const [pendingReferenceRequest, setPendingReferenceRequest] = useState(0);
  const [highlightedReference, setHighlightedReference] = useState<string | null>(null);
  const highlightTimer = useRef<number | null>(null);


  // ??????????????????
  useEffect(() => {
    setStartPage(1);
    setPageInput("");
    setPendingJumpPage(null);
    setPendingReference(null);
    setPendingReferencePage(null);
    setHighlightedReference(null);
    if (highlightTimer.current !== null) {
      window.clearTimeout(highlightTimer.current);
      highlightTimer.current = null;
    }
    setReferenceStartPage(null);
    setPaper(null);
    setLoading(true);
    setPdfUrl(null);
    setShowPdf(false);
    setTranslations({});
    setTranslating(null);
    setIsBatchTranslating(false);
    setQuestion("这篇论文的核心贡献是什么？");
    setAnswer("");
    setSteps([]);
    setSources([]);
    setError("");
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    apiFetch(`/api/papers/${id}/download`)
      .then(async (res) => {
        if (!res.ok) throw new Error("原版 PDF 加载失败");
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) setPdfUrl(objectUrl);
        else URL.revokeObjectURL(objectUrl);
      })
      .catch(() => { if (!cancelled) setPdfUrl(null); });



  return () => { cancelled = true; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch(`/api/papers/${id}?start_page=${startPage}&page_size=${pageSize}`)
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "论文加载失败");
        if (!cancelled) setPaper(data);
      })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "论文加载失败"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [id, startPage]);

  // Wait for the requested page group to load, then align both reading panes.
  useEffect(() => {
    if (loading || showPdf || pendingJumpPage === null || !paper) return;

    const exists = paper.pages?.some(
      (page) => page.page === pendingJumpPage
    );
    if (!exists) return;

    const leftTarget = document.getElementById(`page-${pendingJumpPage}`);
    const rightTarget = document.getElementById(`right-page-${pendingJumpPage}`);

    if (!leftTarget || !rightTarget) return;

    syncPauseUntil.current = Date.now() + 1500;
    syncTarget.current = null;

    if (syncResetTimer.current !== null) {
      window.clearTimeout(syncResetTimer.current);
      syncResetTimer.current = null;
    }

    leftTarget.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

    rightTarget.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

    setPendingJumpPage(null);
  }, [pendingJumpPage, loading, paper, showPdf]);

  const syncScroll = (event: React.UIEvent<HTMLDivElement>, otherId: string) => {
    const source = event.currentTarget;

    // ????????????????????????
    if (Date.now() < syncPauseUntil.current) {
      syncPauseUntil.current = Date.now() + 200;
      return;
    }

    // ?????????????????????????
    if (
      source === syncTarget.current &&
      Math.abs(source.scrollTop - syncExpectedTop.current) < 2
    ) {
      syncTarget.current = null;

      if (syncResetTimer.current !== null) {
        window.clearTimeout(syncResetTimer.current);
        syncResetTimer.current = null;
      }
      return;
    }

    const other = document.getElementById(otherId);
    if (!other || other === source) return;

    const sourceMax = Math.max(0, source.scrollHeight - source.clientHeight);
    const targetMax = Math.max(0, other.scrollHeight - other.clientHeight);
    const ratio = sourceMax > 0 ? source.scrollTop / sourceMax : 0;
    const targetTop = Math.max(0, Math.min(targetMax, ratio * targetMax));

    syncTarget.current = other;
    syncExpectedTop.current = targetTop;

    if (syncResetTimer.current !== null) {
      window.clearTimeout(syncResetTimer.current);
    }

    // ?????????????? scroll ??????????
    syncResetTimer.current = window.setTimeout(() => {
      if (syncTarget.current === other) {
        syncTarget.current = null;
      }
      syncResetTimer.current = null;
    }, 150);

    other.scrollTo({ top: targetTop, behavior: "instant" });
  };

  const translatePage = async (page: Page) => {
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setTranslating(page.page); setError("");
    try {
      const res = await apiFetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: page.text }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "翻译失败");
      setTranslations((prev) => ({ ...prev, [page.page]: data.translation }));
    } catch (e) { setError(e instanceof Error ? e.message : "翻译失败"); }
    finally { setTranslating(null); }
  };

  const translateAll = async () => {
    if (!localStorage.getItem("yantu_token")) {
      window.dispatchEvent(new Event("open-login-modal"));
      return;
    }

    if (isBatchTranslating || !paper) return;

    setIsBatchTranslating(true);
    setError("");

    const allPages: Page[] = [];
    const failedPages: number[] = [];

    try {
      for (let start = 1; start <= paper.page_count; start += pageSize) {
        const res = await apiFetch(
          `/api/papers/${id}?start_page=${start}&page_size=${pageSize}`
        );

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || `第 ${start} 页附近加载失败`);
        }

        if (!Array.isArray(data.pages)) {
          throw new Error(`第 ${start} 页附近数据格式错误`);
        }

        allPages.push(...data.pages);
      }

      const uniquePages = Array.from(
        new Map(allPages.map(p => [p.page, p])).values()
      ).sort((a, b) => a.page - b.page);

      const pagesToTranslate = uniquePages.filter(
        p => p.text?.trim() && !translations[p.page]
      );

      if (pagesToTranslate.length === 0) {
        setError("所有页面均已翻译，无需重复翻译。");
        return;
      }

      for (let i = 0; i < pagesToTranslate.length; i++) {
        const page = pagesToTranslate[i];

        try {
          const res = await apiFetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: page.text })
          });

          const data = await res.json();

          if (!res.ok) {
            throw new Error(data.detail || "翻译请求失败");
          }

          if (typeof data.translation !== "string" || !data.translation.trim()) {
            throw new Error("翻译结果为空");
          }

          setTranslations(prev => ({
            ...prev,
            [page.page]: data.translation
          }));

        } catch (e) {
          console.error(`第 ${page.page} 页翻译失败:`, e);
          failedPages.push(page.page);
        }

        if (i < pagesToTranslate.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

      if (failedPages.length > 0) {
        setError(
          `全文翻译完成，但以下页面失败：${failedPages.join("、")}。可点击对应页面的翻译本页重试。`
        );
      } else {
        setError(`全文翻译完成，共翻译 ${pagesToTranslate.length} 页。`);
      }

    } catch (e) {
      setError(e instanceof Error ? e.message : "全文翻译失败");
    } finally {
      setIsBatchTranslating(false);
    }
  };

  const askPaper = async () => {
    if (!question.trim() || asking) return;
    if (!localStorage.getItem("yantu_token")) { window.dispatchEvent(new Event("open-login-modal")); return; }
    setAsking(true); setAnswer(""); setSteps([]); setSources([]); setError("");
    try {
      const res = await apiFetch("/api/qa/stream", { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify({ paper_id: id, question }) });
      await consumeSSE(res, (data) => {
        if (data.agent) setSteps((prev) => {
          const idx = prev.findIndex((x) => x.agent === data.agent);
          return idx >= 0 ? prev.map((x, i) => i === idx ? { ...x, ...data } : x) : [...prev, data];
        });
        if (data.text) setAnswer((prev) => prev + data.text);
        if (data.sources) setSources(data.sources);
      });
    } catch (e) { setError(e instanceof Error ? e.message : "问答失败"); }
    finally { setAsking(false); }
  };

  const pages = paper?.pages ?? [];

  useEffect(() => {
    if (
      pendingReference === null ||
      pendingReferencePage === null ||
      !paper
    ) return;

    const elementId =
      `${pendingReferenceSide}-ref-${pendingReferencePage}-${pendingReference}`;

    const el = document.getElementById(elementId);
    if (!el) return;

    // ???????????????
    syncPauseUntil.current = Date.now() + 200;
    syncTarget.current = null;
    if (syncResetTimer.current !== null) {
      window.clearTimeout(syncResetTimer.current);
      syncResetTimer.current = null;
    }
    el.scrollIntoView({ behavior: "smooth", block: "center" });

    setHighlightedReference(
      `${pendingReferenceSide}-${pendingReferencePage}-${pendingReference}`
    );

    if (highlightTimer.current !== null) {
      window.clearTimeout(highlightTimer.current);
    }

    highlightTimer.current = window.setTimeout(() => {
      setHighlightedReference(null);
      highlightTimer.current = null;
    }, 5000);

    setPendingReference(null);
    setPendingReferencePage(null);
  }, [
    pendingReference,
    pendingReferencePage,
    pendingReferenceSide,
    pendingReferenceRequest,
    paper,
    startPage,
    translations
  ]);

  const jumpToReference = async (
    rawNum: string,
    side: "left" | "right" = "left"
  ) => {
    const num = Number(rawNum);
    if (!Number.isInteger(num) || num < 1 || !paper) return;

    setError("");

    try {
      let referencesStarted = false;
      let foundReferenceStartPage: number | null = null;
      let targetPage: number | null = null;
      let targetPageData: Page | null = null;

      for (
        let start = 1;
        start <= paper.page_count;
        start += pageSize
      ) {
        let batch: Page[];

        if (start === startPage && paper.pages?.length) {
          batch = paper.pages;
        } else {
          const res = await apiFetch(
            `/api/papers/${id}?start_page=${start}&page_size=${pageSize}`
          );

          const data = await res.json();

          if (!res.ok) {
            throw new Error(data.detail || "加载论文页面失败");
          }

          batch = Array.isArray(data.pages) ? data.pages : [];
        }

        for (const page of batch) {
          const lines = (page.text || "").split(/\r?\n/);

          for (const line of lines) {
            if (
              /(?:^|\n)\s*(\u53c2\u8003\u6587\u732e|\u53c2\u8003\u8d44\u6599|References|Bibliography)\s*[:\uFF1A]?\s*(?:\n|$)/i.test(line)
            ) {
              referencesStarted = true;

              if (foundReferenceStartPage === null) {
                foundReferenceStartPage = page.page;
              }

              continue;
            }

            if (!referencesStarted) continue;

            const entry = line.match(
              /^\s*(?:\*\*)?\[(\d+)\](?:\*\*)?(?:\s|$)/
            );

            if (entry && Number(entry[1]) === num) {
              targetPage = page.page;
              targetPageData = page;
              break;
            }
          }

          if (targetPage !== null) break;
        }

        if (targetPage !== null) break;
      }

      // ?????????????? [n] ???????
      if (targetPage === null || !targetPageData) {
        setError(
          `未找到参考文献 [${num}]，请检查论文解析结果。`
        );
        return;
      }

      // ??????????????????
      if (side === "right" && !translations[targetPage]) {
        if (!localStorage.getItem("yantu_token")) {
          window.dispatchEvent(new Event("open-login-modal"));
          return;
        }

        const translateRes = await apiFetch("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: targetPageData.text })
        });

        const translateData = await translateRes.json();

        if (!translateRes.ok) {
          throw new Error(translateData.detail || "参考文献翻译失败");
        }

        if (
          typeof translateData.translation !== "string" ||
          !translateData.translation.trim()
        ) {
          throw new Error("翻译接口未返回有效译文");
        }

        setTranslations(prev => ({
          ...prev,
          [targetPage!]: translateData.translation
        }));
      }

      setReferenceStartPage(foundReferenceStartPage ?? targetPage);
      setPendingReferenceSide(side);
      setPendingReference(num);
      setPendingReferencePage(targetPage);
      setPendingReferenceRequest(value => value + 1);

      const targetStart =
        Math.floor((targetPage - 1) / pageSize) * pageSize + 1;

      setStartPage(targetStart);
      setPageInput(String(targetPage));
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "参考文献跳转失败"
      );
    }
  };

  const jumpToPage = () => {
    if (!paper) return;

    const page = Number(pageInput);

    if (
      !Number.isInteger(page) ||
      page < 1 ||
      page > paper.page_count
    ) {
      setError(`请输入 1 到 ${paper.page_count} 之间的有效页码`);
      return;
    }

    const targetStart =
      Math.floor((page - 1) / pageSize) * pageSize + 1;

    setError("");
    setPageInput(String(page));
    setShowPdf(false);
    setPendingJumpPage(page);
    setStartPage(targetStart);
  };

  const prevDisabled = startPage <= 1;
  const nextDisabled = !paper || startPage + pageSize > paper.page_count;

  if (loading && !paper) return <div className="p-10 text-center text-sm text-slate-500"><Loader2 className="mx-auto mb-3 h-5 w-5 animate-spin" />正在解析论文内容…</div>;
  if (!paper) return <div className="p-10 text-sm text-rose-700">{error || "论文不存在"}</div>;

  return (
    <div className="flex h-full min-h-[600px] flex-col gap-3 bg-slate-50 p-3 md:p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3">
        <div className="min-w-0"><h1 className="truncate font-semibold text-slate-900">{paper.metadata?.title || paper.filename}</h1><div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">{paper.filename} · 共 {paper.page_count} 页</span>
          <span className="rounded-md bg-slate-100 px-3 py-1 text-sm text-slate-600">
            第 {startPage}-{Math.min(startPage + pages.length - 1, paper.page_count)} 页
          </span>
        </div></div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="number"
            aria-label="跳转到指定页码"
            min={1}
            max={paper.page_count}
            value={pageInput}
            onChange={(event) => setPageInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                jumpToPage();
              }
            }}
            placeholder="页码"
            className="w-20 rounded-lg border border-slate-200 px-2 py-2 text-xs text-slate-700"
          />
          <button
            type="button"
            onClick={jumpToPage}
            disabled={loading}
            className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:opacity-40"
          >
            跳转
          </button>
          <select aria-label="章节导航" defaultValue="" onChange={(event) => { const section = paper.metadata?.sections?.find((item) => item.title === event.target.value); if (!section) return; const pageStart = Math.floor((section.page - 1) / pageSize) * pageSize + 1; setStartPage(pageStart); setPageInput(String(section.page)); window.setTimeout(() => document.getElementById(`page-${section.page}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 220); }} className="max-w-56 rounded-lg border border-slate-200 bg-white px-2 py-2 text-xs text-slate-600"><option value="">章节快速导航</option>{(paper.metadata?.sections || []).map((section, index) => <option key={`${section.page}-${index}`} value={section.title}>{"　".repeat(Math.max(0, section.level - 1))}{section.title} · p.{section.page}</option>)}</select><button disabled={prevDisabled || loading} onClick={() => { const next = Math.max(1, startPage - pageSize); setStartPage(next); setPageInput(String(next)); }} className="rounded-lg border border-slate-200 p-2 text-slate-600 disabled:opacity-30"><ChevronLeft className="h-4 w-4" /></button><button disabled={nextDisabled || loading} onClick={() => { const next = startPage + pageSize; setStartPage(next); setPageInput(String(next)); }} className="rounded-lg border border-slate-200 p-2 text-slate-600 disabled:opacity-30"><ChevronRight className="h-4 w-4" /></button></div>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_360px]">
        <section className="flex min-h-[420px] min-w-0 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-700"><span className="flex items-center gap-2"><BookOpen className="h-4 w-4 text-blue-600" /> 原文解析</span><div className="flex gap-1"><button onClick={() => setShowPdf(false)} className={`rounded-md px-2 py-1 text-xs ${!showPdf ? "bg-blue-100 text-blue-700" : "text-slate-500 hover:bg-slate-100"}`}>解析文本</button><button onClick={() => setShowPdf(true)} disabled={!pdfUrl} className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs disabled:opacity-40 ${showPdf ? "bg-blue-100 text-blue-700" : "text-slate-500 hover:bg-slate-100"}`}><FileText className="h-3 w-3" /> PDF 原版</button></div></div>
          {showPdf ? <iframe title="论文 PDF 原版版式" src={pdfUrl || undefined} className="min-h-0 flex-1 bg-slate-100" /> : <div id="left" onScroll={(e) => syncScroll(e, "right")} className="min-h-0 flex-1 space-y-7 overflow-auto p-5 text-sm leading-7 text-slate-700">
            {pages.map((page) => {
              const heading = (page.text || "").match(/(?:^|\n)\s*(\u53c2\u8003\u6587\u732e|\u53c2\u8003\u8d44\u6599|References|Bibliography)\s*[:\uFF1A]?\s*(?:\n|$)/i);
              const isReferencesPage =
                referenceStartPage !== null && page.page >= referenceStartPage;

              return (
                <article id={`page-${page.page}`} key={page.page}>
                  <div className="mb-2 flex items-center gap-2">
                    <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">第 {page.page} 页</span>
                    {page.method === "ocr" && <span className="text-[10px] text-amber-600">OCR 识别</span>}
                  </div>
                  <div className="whitespace-pre-wrap font-serif">
                    {page.text ? (
                      heading && heading.index !== undefined ? (
                        <>
                          {page.text.slice(0, heading.index) ? renderCitations(page.text.slice(0, heading.index), (num) => jumpToReference(num, 'left')) : null}
                          <div className="mt-4 border-t border-slate-200 pt-3">
                            <h3 className="text-lg font-bold mb-3">{heading[1]}</h3>
                            {renderReferences(page.text.slice(heading.index + heading[0].length), 'left', page.page, highlightedReference)}
                          </div>
                        </>
                      ) : isReferencesPage ? (
                        renderReferences(page.text, 'left', page.page, highlightedReference)
                      ) : (
                        renderCitations(page.text, (num) => jumpToReference(num, 'left'))
                      )
                    ) : "暂无可显示的文本内容"}
                  </div>
                </article>
              );
            })}
          </div>}
        </section>

        <section className="flex min-h-[420px] min-w-0 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="flex items-center justify-between border-b bg-slate-50 px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-700"><Languages className="h-4 w-4 text-emerald-600" /> 对照翻译</div>
            <button onClick={() => void translateAll()} disabled={isBatchTranslating || pages.length === 0} className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-50 transition">
              {isBatchTranslating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Languages className="h-3.5 w-3.5" />}
              {isBatchTranslating ? "流水线翻译中..." : "一键全文翻译"}
            </button>
          </div>
          <div id="right" onScroll={(e) => syncScroll(e, "left")} className="min-h-0 flex-1 space-y-7 overflow-auto p-5 text-sm leading-7 text-slate-700">
            {pages.map((page) => <article id={`right-page-${page.page}`} key={page.page}><div className="mb-2 flex items-center justify-between"><span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">第 {page.page} 页</span><button disabled={translating === page.page} onClick={() => void translatePage(page)} className="inline-flex items-center gap-1 rounded-md border border-blue-200 px-2 py-1 text-xs text-blue-700 hover:bg-blue-50 disabled:opacity-50">{translating === page.page ? <Loader2 className="h-3 w-3 animate-spin" /> : null}{translations[page.page] ? "重新翻译" : "翻译本页"}</button></div><div className="whitespace-pre-wrap">{translations[page.page] ? renderTranslatedContent(
              translations[page.page],
              (num) => jumpToReference(num, 'right'),
              page.page,
              referenceStartPage !== null && page.page >= referenceStartPage,
              highlightedReference
            ) : "点击“翻译本页”生成对照译文。"}</div></article>)}
          </div>
        </section>

        <aside className="flex min-h-[420px] min-w-0 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="flex items-center gap-2 border-b bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800"><Bot className="h-4 w-4 text-blue-600" /> 论文问答与证据溯源</div>
          <div className="min-h-0 flex-1 space-y-3 overflow-auto p-4">
            {steps.map((step, i) => (
              <div key={`${step.agent}-${i}`} className="rounded-lg border border-slate-100 p-2">
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  {step.status === "running" ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600" />
                  ) : step.status === "error" ? (
                    <span className="font-bold text-rose-600">?</span>
                  ) : step.status === "warning" ? (
                    <span className="font-bold text-amber-600">!</span>
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                  )}
                  <span className="font-medium">{step.agent}</span>
                  <span className={step.status === "error" ? "text-rose-600" : step.status === "warning" ? "text-amber-600" : "text-slate-400"}>
                    {step.status === "running" ? "进行中" : step.status === "error" ? "失败" : step.status === "warning" ? "警告" : step.status === "done" ? "完成" : ""}
                  </span>
                </div>
                {step.validation && (
                  <div className="mt-2 space-y-2 text-xs text-slate-600">
                    {step.validation.message && <p>{step.validation.message}</p>}
                    {step.validation.errors?.length > 0 && (
                      <ul className="list-disc pl-5 text-rose-600">
                        {step.validation.errors.map((item: string, j: number) => <li key={j}>{item}</li>)}
                      </ul>
                    )}
                    {step.validation.warnings?.length > 0 && (
                      <ul className="list-disc pl-5 text-amber-600">
                        {step.validation.warnings.map((item: string, j: number) => <li key={j}>{item}</li>)}
                      </ul>
                    )}
                    {step.validation.semantic_support_checked === false && (
                      <p className="text-amber-600">{"\u5c1a\u672a\u6267\u884c\u8bed\u4e49\u652f\u6301\u6821\u9a8c\uff1b\u5f53\u524d\u4ec5\u68c0\u67e5\u5f15\u7528\u7f16\u53f7\uff0c\u4e0d\u80fd\u636e\u6b64\u786e\u8ba4\u56de\u7b54\u5185\u5bb9\u5f97\u5230\u539f\u6587\u652f\u6301\u3002"}</p>
                    )}
                    {step.validation.semantic_check?.semantic_support_checked === true && (
                      <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-slate-800">{"\u8bc1\u636e\u652f\u6301\u6838\u9a8c"}</span>
                          <span className={`rounded-full px-2 py-1 font-semibold ${
                            step.validation.semantic_check.valid
                              ? "bg-emerald-50 text-emerald-700"
                              : "bg-amber-50 text-amber-700"
                          }`}>
                            {step.validation.semantic_check.claim_count ?? 0}{" \u6761\u4e3b\u5f20\u83b7\u652f\u6301"}
                          </span>
                        </div>
                        {step.validation.semantic_check.claims?.length > 0 ? (
                          <div className="space-y-3">
                            {step.validation.semantic_check.claims.map((claim: any, j: number) => {
                              const verdictLabels: Record<string, string> = {
                                supported: "\u652f\u6301",
                                partially_supported: "\u90e8\u5206\u652f\u6301",
                                unsupported: "\u4e0d\u652f\u6301",
                                insufficient_evidence: "\u8bc1\u636e\u4e0d\u8db3",
                              };
                              const verdictStyles: Record<string, string> = {
                                supported: "bg-emerald-50 text-emerald-700",
                                partially_supported: "bg-amber-50 text-amber-700",
                                unsupported: "bg-rose-50 text-rose-700",
                                insufficient_evidence: "bg-slate-100 text-slate-700",
                              };
                              const verdict = claim.verdict || "unknown";
                              return (
                                <div key={`${j}-${claim.claim}`} className="border-t border-slate-100 pt-3 first:border-0 first:pt-0">
                                  <div className="mb-1 flex flex-wrap items-start justify-between gap-2">
                                    <span className="min-w-0 flex-1 font-medium text-slate-800">
                                      {j + 1}. {claim.claim}
                                    </span>
                                    <span className={`shrink-0 rounded-full px-2 py-1 font-semibold ${verdictStyles[verdict] || "bg-slate-100 text-slate-600"}`}>
                                      {verdictLabels[verdict] || "\u672a\u8bc6\u522b"}
                                    </span>
                                  </div>
                                  {claim.evidence_ids?.length > 0 && (
                                    <p className="mb-1 text-blue-700">
                                      {"\u5bf9\u5e94\u8bc1\u636e\uff1a"}{claim.evidence_ids.join("\u3001")}
                                    </p>
                                  )}
                                  {claim.reason && (
                                    <p className="leading-5 text-slate-500">
                                      {"\u5224\u5b9a\u7406\u7531\uff1a"}{claim.reason}
                                    </p>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <p className="text-amber-600">{"\u6821\u9a8c\u5df2\u6267\u884c\uff0c\u4f46\u6ca1\u6709\u8fd4\u56de\u9010\u6761\u4e3b\u5f20\u7ed3\u679c\u3002"}</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {answer && (
              <div className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm leading-7 text-slate-800">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    h1: ({ ...props }) => <h1 className="mb-3 mt-4 text-xl font-bold text-slate-900" {...props} />,
                    h2: ({ ...props }) => <h2 className="mb-2 mt-4 text-lg font-bold text-slate-900" {...props} />,
                    h3: ({ ...props }) => <h3 className="mb-2 mt-3 text-base font-bold text-slate-900" {...props} />,
                    p: ({ ...props }) => <p className="my-2" {...props} />,
                    ul: ({ ...props }) => <ul className="my-2 list-disc space-y-1 pl-6" {...props} />,
                    ol: ({ ...props }) => <ol className="my-2 list-decimal space-y-1 pl-6" {...props} />,
                    li: ({ ...props }) => <li className="pl-1" {...props} />,
                    strong: ({ ...props }) => <strong className="font-bold text-slate-900" {...props} />,
                    blockquote: ({ ...props }) => <blockquote className="my-3 border-l-4 border-blue-300 pl-3 text-slate-600" {...props} />,
                    table: ({ ...props }) => <div className="my-3 overflow-x-auto"><table className="w-full border-collapse text-left text-xs" {...props} /></div>,
                    th: ({ ...props }) => <th className="border border-slate-300 bg-slate-100 px-2 py-1 font-semibold" {...props} />,
                    td: ({ ...props }) => <td className="border border-slate-300 px-2 py-1" {...props} />,
                    pre: ({ ...props }) => <pre className="my-3 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100" {...props} />,
                    code: ({ ...props }) => <code className="rounded bg-slate-200 px-1 py-0.5 text-[0.9em]" {...props} />,
                    hr: ({ ...props }) => <hr className="my-4 border-slate-300" {...props} />,
                  }}
                >
                  {answer}
                </ReactMarkdown>
              </div>
            )}
            {sources.length > 0 && <div><h3 className="mb-2 text-xs font-semibold text-slate-500">原文溯源</h3>{sources.map((source, i) => <div key={source.chunk_id || i} className="mb-2 rounded-lg border border-slate-200 border-l-2 border-l-blue-500 bg-slate-50 p-3 text-xs"><div className="mb-1 font-semibold text-slate-700">{source.evidence_id} · 第 {source.page} 页</div><p className="leading-5 text-slate-600">{source.quote}</p></div>)}</div>}
            {error && <p className="rounded-lg bg-rose-50 p-2 text-xs text-rose-700">{error}</p>}
          </div>
          <div className="border-t p-3"><input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void askPaper()} placeholder="向这篇论文提问…" className="mb-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-500" /><button onClick={() => void askPaper()} disabled={asking || !question.trim()} className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{asking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} 提问</button></div>
        </aside>
      </div>
    </div>
  );
}
