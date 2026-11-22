import { Agent } from "undici";

/**
 * Same-origin streaming proxy to FastAPI.
 * Set BACKEND_URL in frontend/.env.local.
 */
const BACKEND_URL =
  process.env.BACKEND_URL || "http://127.0.0.1:8000";

const UPLOAD_DISPATCHER = new Agent({
  headersTimeout: 20 * 60 * 1000,
  bodyTimeout: 20 * 60 * 1000,
});

async function proxy(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path = [] } = await context.params;
  const safePath = path.map((part) => encodeURIComponent(part)).join("/");
  const incoming = new URL(request.url);
  const target =
    `${BACKEND_URL.replace(/\/$/, "")}/api/${safePath}${incoming.search}`;

  const headers = new Headers();

  for (const key of [
    "authorization",
    "content-type",
    "accept",
    "x-request-id",
  ]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }

  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";

  try {
    const upstream = await fetch(target, {
      method,
      headers,
      // Forward the incoming stream instead of buffering the whole upload.
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch when forwarding a ReadableStream body.
      ...(hasBody ? { duplex: "half" as const } : {}),
      cache: "no-store",
      signal: request.signal,
      ...(safePath === "papers/upload"
        ? { dispatcher: UPLOAD_DISPATCHER }
        : {}),
    } as RequestInit & {
      dispatcher?: Agent;
      duplex?: "half";
    });

    const responseHeaders = new Headers();

    for (const key of [
      "content-type",
      "cache-control",
      "content-disposition",
      "x-request-id",
    ]) {
      const value = upstream.headers.get(key);
      if (value) responseHeaders.set(key, value);
    }

    responseHeaders.set("X-Accel-Buffering", "no");

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("FastAPI proxy error:", error);

    return Response.json(
      {
        detail:
          "后端服务不可用，请确认 FastAPI 已启动且 BACKEND_URL 配置正确。",
      },
      { status: 502 },
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
