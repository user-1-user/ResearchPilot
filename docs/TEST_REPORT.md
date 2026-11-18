# ResearchPilot 测试报告

## 1. 测试目标

验证 ResearchPilot 核心功能、后端接口、检索问答流程及部署配置的基本可用性。

## 2. 测试环境

- 操作系统：Windows 11
- 后端：Python 3.11、FastAPI
- 前端：Next.js 16、React 19
- 数据库：SQLite
- 部署：Docker Compose
- 大模型：DeepSeek API
- 向量模型：BGE-M3
- 重排序模型：BGE Reranker

## 3. 后端自动化测试

执行命令：

```powershell
python -m unittest tests.test_evidence_validation -v
```

测试结果：

- 证据结构校验：PASS
- 引用 ID 校验：PASS
- 语义支持校验：PASS
- supported 但缺少有效证据 ID 的情况：PASS

共 14 项测试通过。

## 4. 后端冒烟测试

执行命令：

```powershell
python tests/smoke_test.py
```

测试时配置了本机 Reranker 模型路径。

测试结果：

| 测试项目 | 结果 |
|---|---|
| 健康检查 | PASS |
| 用户注册与登录 | PASS |
| PDF 上传 | PASS |
| 重复文献检测 | PASS |
| 分页查询 | PASS |
| 文献检索 | PASS |
| QA SSE | PASS |
| 多论文对比 SSE | PASS |
| 知识图谱 | PASS |
| Future Work | PASS |
| 文献综述 | PASS |
| Agent 工具调用 | PASS |

最终结果：`ResearchPilot smoke test: PASS`

## 5. 前端构建测试

执行命令：

```powershell
npm run build
```

测试结果：

- Next.js 编译：PASS
- 页面生成：PASS
- 共生成 9 个页面
- 构建命令正常结束

说明：本次构建日志显示跳过了类型验证，因此不将 TypeScript 类型检查单独标记为通过。

## 6. 证据验证专项修复

发现的问题：

语义审核模型可能返回 `supported`，但没有提供有效证据 ID。

修复措施：

- 对 supported 结果增加有效 evidence ID 检查。
- 缺少有效证据 ID 时，将结果降级为 `insufficient_evidence`。
- 增加专项单元测试。
- Reader 页面增加验证失败或未执行时的风险提示。

专项测试结果：14 项测试全部通过。

## 7. Docker 部署测试

已完成的验证：

- Docker 镜像构建：PASS
- Docker Compose 配置检查：PASS
- 容器启动及模型挂载：PASS
- 后端健康检查：PASS
- 容器内模型加载及向量推理：PASS
- 前端通过代理访问后端：PASS

## 8. 已知限制

1. 语义支持判断仍依赖独立 LLM 审核，不能完全保证证据与每条 Claim 在语义上严格对应。
2. 当前前端构建日志跳过了 TypeScript 类型验证。
3. 尚未完成全新电脑、干净环境下的跨机器部署验收。
4. 尚未完成公网部署和评审账号验收。

## 9. 测试结论

已完成核心后端接口、证据验证、前端构建及本机 Docker 部署的主要测试。

当前测试结果支持项目核心功能在本机环境运行。跨机器部署、公网访问及完整比赛演示仍需进一步验收。