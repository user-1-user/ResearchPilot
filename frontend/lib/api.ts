/** Authenticated same-origin API helper for the Next.js -> FastAPI proxy. */
export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  const headers = new Headers(init.headers || {});
  if (typeof window !== "undefined") {
    const token = window.localStorage.getItem("yantu_token");
    if (token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }
  return fetch(input, { ...init, headers, cache: "no-store" });
}

/** Correctly handles SSE frames split across arbitrary network chunks. */
export async function consumeSSE(
  response: Response,
  onMessage: (data: any) => void,
): Promise<void> {
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // The upstream may have returned plain text.
    }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("服务器没有返回流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        for (const line of frame.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          try {
            onMessage(JSON.parse(raw));
          } catch {
            // Ignore malformed keepalive frames instead of breaking the stream.
          }
        }
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      for (const line of buffer.split(/\r?\n/)) {
        if (line.startsWith("data:")) {
          try { onMessage(JSON.parse(line.slice(5).trim())); } catch { /* ignore */ }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
