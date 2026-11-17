# ResearchPilot 检索与重排序实验报告（V6—V8）

## 一、实验目的

本系列实验用于评估 ResearchPilot 文献检索系统中混合检索与 CrossEncoder 重排序策略的效果，并探索在降低推理成本的同时尽可能保持检索质量的 Pre-CrossEncoder Gate 方案。

核心研究问题：

1. BM25 与 Dense 混合检索能否提升相关文献片段的召回质量？
2. CrossEncoder 全量重排序能否进一步改善相关片段的排序？
3. Selective Rerank 是否能够在质量与计算成本之间取得更好的平衡？
4. Frozen Gate 能否减少 CrossEncoder 调用，同时保持可接受的检索质量？

## 二、实验方法与评价指标

### 2.1 检索流程

正式系统采用以下检索流程：

1. BM25：计算查询与文献片段之间的词汇匹配相关性。
2. Dense Retrieval：计算查询向量与文献片段向量之间的语义相似度。
3. RRF：通过 Reciprocal Rank Fusion 融合稀疏检索与稠密检索的排序结果。
4. CrossEncoder：对初筛候选片段进行相关性评分，并重新排序。

RRF 使用排名融合，避免直接混合 BM25 与 Dense 两种不同量纲的原始分数。

### 2.2 评价指标

- Recall@5：前 5 个结果中是否覆盖相关证据的召回比例。
- Recall@10：前 10 个结果中是否覆盖相关证据的召回比例。
- MRR：第一个相关结果排名倒数的平均值，越高表示相关证据越靠前。
- Rank-1 Hit：第一名结果是否命中相关证据。
- CrossEncoder 调用率：实际需要执行 CrossEncoder 重排序的问题比例。

以上指标均按照各实验数据集及其对应评估脚本的定义统计。

## 三、V6：三种检索策略对比实验

### 3.1 实验设置

- 问题数量：80
- 文献数量：4 篇
- 对比策略：BASELINE、FULL_HYBRID、SELECTIVE_HYBRID
- 三种策略使用相同的 80 道问题，候选数量分布完全一致。
- 每种策略均包含 80 条评估记录，共 240 条记录。

候选数量分布：

| 候选数量 | 问题数量 |
|---|---:|
| 24 | 20 |
| 26 | 20 |
| 30 | 20 |
| 57 | 1 |
| 58 | 19 |

三种策略的候选数量分布完全一致，保证了候选池规模方面的对照公平性。

### 3.2 实验结果

| 指标 | BASELINE | FULL_HYBRID | SELECTIVE_HYBRID |
|---|---:|---:|---:|
| Recall@5 | 0.6813 | 0.7563 | 0.6813 |
| Recall@10 | 0.8625 | 0.8938 | 0.8625 |
| MRR | 0.4335 | 0.5120 | 0.4597 |
| Rank-1 Hit | 20 | 25 | 23 |
| Top-5 Hit | 56 | 63 | 56 |

### 3.3 结果分析

FULL_HYBRID 相比 BASELINE：

- Recall@5 提升 0.0750。
- Recall@10 提升 0.0313。
- MRR 提升约 0.0784。
- Rank-1 Hit 从 20 增加到 25。

SELECTIVE_HYBRID 相比 BASELINE：

- MRR 从 0.4335 提升至 0.4597。
- Rank-1 Hit 从 20 增加到 23。
- Recall@5 和 Recall@10 没有提升。

FULL_HYBRID 在本次实验的主要检索指标上表现最好。SELECTIVE_HYBRID 虽然改善了部分排序指标，但没有取得与 FULL_HYBRID 相同的召回效果。

现有实验报告记录的 V6 MRR 增益 Bootstrap 95% 置信区间为 [0.0490, 0.1112]。该区间支持本次样本中 FULL_HYBRID 相对 BASELINE 的 MRR 改善，但不能据此推断对所有文献和问题都具有相同效果。

## 四、V7：跨文献检索评估

### 4.1 实验设置

- 问题数量：80
- 文献数量：4 篇
- Chunk 数量：138
- 对比策略：BASELINE 与 HYBRID
- 所有问题均在包含 4 篇文献的同一全局语料库中检索。

### 4.2 实验结果

以下采用 v7_global_summary 中的宏平均指标：

| 指标 | BASELINE | HYBRID | 提升 |
|---|---:|---:|---:|
| Recall@5 | 0.4625 | 0.6125 | +0.1500 |
| Recall@10 | 0.6250 | 0.6813 | +0.0563 |
| MRR | 0.3319 | 0.4215 | +0.0896 |

### 4.3 结果分析

HYBRID 相比 BASELINE，在 Recall@5、Recall@10 和 MRR 三项指标上均有提升。

其中 MRR 提升约 0.0896，说明相关证据的排序位置得到改善。

补充说明：v7_combined_gate_results.json 还记录了 Gate 规则探索结果，其中包含 BASELINE 和 FULL_HYBRID 的对照指标。这些数据属于规则探索实验，不作为本节 V7 主实验的宏平均结果。

V7 与 V6 的数据集及候选集设置并不完全相同，因此不应将两组绝对指标直接作为严格的横向性能比较。
## 五、V8：Frozen Gate 独立评估

### 5.1 实验设置

- 问题数量：39
- 文献数量：2 篇
- 对比策略：BASELINE、FULL CROSSENCODER、FROZEN GATE。

Frozen Gate 使用预先冻结的判断规则：

当以下条件同时成立时，触发 CrossEncoder：

1. BM25 Top-1 在 Dense 排名中的位置大于 10，或者 Dense Top-1 在 BM25 排名中的位置大于 10。
2. RRF Top-1 的两路排名差距大于 1。

该规则在本次评估中保持冻结，没有根据这 39 道测试问题重新调整阈值。

### 5.2 实验结果

| 指标 | BASELINE | FULL CROSSENCODER | FROZEN GATE |
|---|---:|---:|---:|
| Recall@5 | 0.6410 | 0.7179 | 0.6667 |
| Recall@10 | 0.7692 | 0.8205 | 0.7692 |
| MRR | 0.5452 | 0.6164 | 0.5813 |
| Rank-1 Hit | 16 | 20 | 18 |

### 5.3 推理成本分析

Frozen Gate 选择 39 道问题中的 8 道执行 CrossEncoder：

- Gate 触发率：20.51%。
- 跳过 CrossEncoder 的问题：31 道。
- 估算 CrossEncoder 调用量减少：79.49%。

注意：该调用量是按照 Gate 部署路径计算的反事实估算值。实验中的 FULL CROSSENCODER 参考组仍然对全部问题执行了 CrossEncoder，不能将 Gate 的估算调用量误认为整个实验实际消耗的推理调用量。

### 5.4 Bootstrap 结果

Gate 相比 FULL CROSSENCODER 的指标差值：

| 指标 | Gate - Full CE | Bootstrap 95% CI |
|---|---:|---:|
| MRR | -0.0351 | [-0.0735, -0.0066] |
| Recall@5 | -0.0492 | [-0.1190, 0] |
| Recall@10 | -0.0492 | [-0.1136, 0] |

本实验采用 5000 次按文献分层的 Cluster Bootstrap。

结果表明，Gate 的 MRR 低于全量 CrossEncoder；Recall@5 和 Recall@10 也没有显示出优于全量重排的结果。

因此，Gate 展现出降低计算调用量的潜力，但目前不能认为它在检索质量上等价于全量 CrossEncoder。

## 六、正式系统采用的重排序方案

根据当前正式代码及 backend/.env 配置检查：

- ENABLE_RERANKER=1。
- RERANKER_MODEL 指向 BAAI bge-reranker-base 本地模型目录。
- RERANK_MIN_SCORE=-2.0。
- DEFAULT_TOP_K=7。
- 普通查询最多构建 28 个候选。
- 满足跨语言扩展条件时，候选数量最多为 60。
- 初筛采用 BM25、Dense 与 RRF。
- 启用 CrossEncoder 对候选片段进行全量重排序。

正式系统暂时保留全量 CrossEncoder，主要原因：

1. V6 中 FULL_HYBRID 的综合检索指标优于 BASELINE 和 SELECTIVE_HYBRID。
2. V8 中 FULL CROSSENCODER 的 MRR、Recall@5 和 Recall@10 均高于 FROZEN GATE。
3. Gate 虽然能够显著减少 CrossEncoder 调用，但仍存在检索质量下降。
4. 当前比赛阶段优先保证证据检索质量，暂不以牺牲检索质量换取推理成本下降。

说明：上述配置来自代码和 .env 文件检查。正式运行时是否成功加载模型，还应结合后端启动日志确认。

## 七、实验结论

综合 V6、V7、V8：

1. 在 V6 和 V7 的实验数据上，混合检索与重排序策略取得了优于各自 BASELINE 的结果。
2. V6 中 FULL_HYBRID 的主要指标优于 SELECTIVE_HYBRID。
3. V8 中 FROZEN GATE 能够减少约 79.49% 的 CrossEncoder 调用，但检索指标低于 FULL CROSSENCODER。
4. 当前正式系统采用 BM25 + Dense + RRF + 全量 CrossEncoder 的质量优先方案。
5. Frozen Gate 暂作为后续性能优化方向，不作为当前正式系统默认策略。

## 八、实验局限性

1. V6 和 V7 均只覆盖 4 篇文献、80 道问题，样本规模有限。
2. V8 仅覆盖 2 篇文献、39 道问题，尚不足以证明 Gate 在更大规模文献库中的普遍表现。
3. 不同版本的数据集、候选生成条件和实验目的存在差异，不能直接将不同版本的绝对指标视为同条件对照。
4. 当前结果主要反映既定数据集上的检索排序表现，不等同于端到端问答准确率、用户满意度或完整系统性能。
5. Gate 的调用量减少是基于冻结规则的部署路径估算，实际部署成本仍需通过运行时延迟、显存占用和模型调用统计进一步验证。
6. 正式系统的运行参数与实验参数并非完全相同，实验结果不能直接等同于线上运行效果。

## 九、后续工作

1. 扩大跨文献测试集，增加文献类型和问题类型。
2. 增加端到端问答正确性、证据引用准确性和拒答能力评估。
3. 记录正式系统的检索延迟、重排延迟、显存和模型调用次数。
4. 在更大规模独立数据集上继续评估 Frozen Gate。
5. 保留 BASELINE、FULL_HYBRID 和 FROZEN GATE 的可复现实验脚本及原始结果。

---

**项目：** ResearchPilot（研途智研）
**报告范围：** V6、V7、V8 检索与重排序实验
**正式策略：** BM25 + Dense + RRF + 全量 CrossEncoder