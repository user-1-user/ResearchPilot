import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

async function proxyRequest(req: NextRequest) {
  const path = req.nextUrl.pathname;
  const method = req.method;

  try {
    const contentType = req.headers.get('content-type') || '';
    // GET 请求没有 body，其他请求读取 body
    const body = method !== 'GET' && method !== 'HEAD' ? await req.arrayBuffer() : undefined;

    const res = await fetch(`${BACKEND_URL}${path}`, {
      method,
      headers: contentType ? { 'Content-Type': contentType } : {},
      body: body,
    });

    // 1. 针对流式问答 (SSE) 的特殊处理
    if (res.headers.get('content-type')?.includes('text/event-stream')) {
      return new NextResponse(res.body, {
        status: res.status,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }

    // 2. 统一读取文本并解析为 JSON（无论是成功还是报错，保证前端能拿到 detail）
    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }

    return NextResponse.json(data, { status: res.status });

  } catch (error) {
    console.error(`代理 ${method} 请求失败:`, error);
    return NextResponse.json({ detail: "无法连接到后端服务" }, { status: 502 });
  }
}

// 导出所有需要的方法
export async function GET(req: NextRequest) { return proxyRequest(req); }
export async function POST(req: NextRequest) { return proxyRequest(req); }
export async function DELETE(req: NextRequest) { return proxyRequest(req); }