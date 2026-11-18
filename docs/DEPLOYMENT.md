# ResearchPilot 部署指南

## 1. 环境要求

- Windows 10/11 或 Linux
- Python 3.11
- Node.js 20+
- Docker Desktop / Docker Engine
- Docker Compose
- Git

## 2. 项目目录

```text
yantu-agent/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .env.example
│   └── tests/
├── frontend/
├── docs/
└── README.md
```

## 3. 配置环境变量

进入 `backend` 目录，将 `.env.example` 复制为 `.env`：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少配置：

- `LLM_API_KEY`：大模型服务 API Key
- `LLM_BASE_URL`：大模型 API 地址
- `LLM_MODEL`：模型名称
- `BGE_M3_HOST_PATH`：本地 BGE-M3 模型目录
- `RERANKER_HOST_PATH`：本地 Reranker 模型目录

模型目录需要提前下载并配置为本机实际路径。不要将 `.env` 或真实 API Key 提交到 Git 仓库。

## 4. Docker 部署

在项目根目录执行：

```powershell
docker compose -f .\backend\docker-compose.yml config -q
docker compose -f .\backend\docker-compose.yml build
docker compose -f .\backend\docker-compose.yml up -d
```

查看容器状态：

```powershell
docker compose -f .\backend\docker-compose.yml ps
```

查看后端日志：

```powershell
docker compose -f .\backend\docker-compose.yml logs -f
```

检查后端健康状态：

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

停止服务：

```powershell
docker compose -f .\backend\docker-compose.yml down
```

注意：不要使用 `docker compose down -v`，否则可能删除持久化数据卷。

## 5. 前端部署

进入 `frontend` 目录，配置 `.env.local`：

```dotenv
BACKEND_URL=http://127.0.0.1:8000
```

安装依赖并启动：

```powershell
npm install
npm run dev
```

浏览器访问：

```text
http://localhost:3000
```

生产构建：

```powershell
npm run build
npm run start
```

## 6. 常见问题

### 6.1 后端无法启动

检查 Docker 容器状态及日志：

```powershell
docker compose -f .\backend\docker-compose.yml ps
docker compose -f .\backend\docker-compose.yml logs --tail 100
```

### 6.2 模型加载失败

检查 `.env` 中的模型宿主机路径是否真实存在，并确认模型文件完整。

### 6.3 大模型 API 调用失败

检查 API Key、Base URL、模型名称及网络连接。

### 6.4 前端无法连接后端

检查 `frontend/.env.local` 中的 `BACKEND_URL`，并确认后端健康检查正常。

## 7. 数据持久化

Docker Compose 使用 `researchpilot-data` 命名卷保存应用数据。普通停止和重新启动不会主动删除该数据卷。

## 8. 安全说明

- 不要将 API Key 提交到公开仓库。
- `.env` 文件仅用于本地部署。
- 对外部署时应配置 HTTPS、访问控制及安全的密钥管理方式。
- 公开演示前应检查日志和仓库，避免泄露密钥及用户数据。