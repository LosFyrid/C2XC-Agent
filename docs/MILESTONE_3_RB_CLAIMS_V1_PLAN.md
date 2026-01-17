# Milestone 3.1：ReasoningBank 质量升级（RBMEM_CLAIMS_V1 + Claim 检索 + PubChem/COOH）

最后更新：2026-01-17

> 本文档记录 **Milestone 3 的“统一升级方案（完整版）”**：把 RB 经验条目升级为严格结构化协议，并将检索单位下沉到 claim，但保持 item 为真值与引用单位（`mem:<id>`）。同时补齐 RB learn 的质量门禁、候选集更新、可解释状态机，以及 PubChem/COOH 可视化审查链路。

---

## 0) 目标（What / Why）

- **经验条目“可机器解析”**：RB memory 不再是自由文本，而是可版本化、可验证、可渲染的结构化协议，避免“写进去就再也读不回”。
- **检索更细粒度**：向量检索的基本单位从“整条 item”下沉到“claim”，避免一条长经验里只有一句有用但被稀释。
- **引用保持稳定**：agent/报告仍然只引用 item：`mem:<uuid>`；claim 只是用于检索/解释的派生索引。
- **RB learn 更可信、更可控**：
  - FACTS 层由系统注入（来自 run/feedback/trace），降低幻觉；
  - learn 只能 update 候选集（candidate memories），防止“全库乱改”；
  - claim 级 support/contra 状态机通用且保守，支持“验证后移动”的语义；
  - 写入前质量门禁 + 一次格式修复重试，避免污染库。
- **COOH 人工审查更高效**：提供 modifier→PubChem CID/SMILES + COOH 子结构判定证据，UI 展示给用户，默认不阻塞 run。

---

## 1) 本轮范围与明确不做（Scope）

### 1.1 本轮要做（In Scope）

- 引入 `RBMEM_CLAIMS_V1` 严格结构化协议（存入 `MemoryItem.content`）。
- Chroma 单 collection 内支持 `doc_type=item|claim`：
  - item doc 为真值（浏览/引用）；
  - claim doc 为派生索引（embedding 检索）。
- 统一检索语义：`mem_search` 与 `/api/v1/memories?query=` 走同一套“搜 claim → 返回 item”的逻辑。
- RB learn 升级：
  - FACTS 系统注入；
  - structured output + 质量门禁；
  - candidate memories 更新范围控制；
  - claim 级状态机（support/contra/irrelevant）。
- PubChem + COOH 审查（方案 2）：后台查询+缓存，UI 展示证据。
- 迁移策略：本机选 **B（重写迁移）**；服务器是测试版可直接删数据，不做线上迁移。

### 1.2 本轮不做（Out of Scope / 明确排除）

- 不做硬约束 #11：不搞 FACTS 的“字段名/数值黑名单”（避免误杀科研表达）。
- 不做 D2：不将 `max_steps/剩余步数` 注入提示词作为硬偏置。
- 不强制 RB learn 继承 run 的 config snapshot（RB learn 与 run 保持可用不同模型/网关的灵活性）。
- 不要求 learn delta 覆盖“archive 生产路径”（仍保留手工 archive 即可）。

---

## 2) 硬约束（Hard Constraints：代码必须 enforce）

> 以下均为“写入前必须满足”的硬约束，避免依赖模型自觉。

1) **Item 生命周期状态只允许两种：`active | archived`**
- `MemoryItem.status` 仅表达生命周期，不表达认知状态。

2) **认知状态必须是 claim-level（同一 item 内允许混合）**
- claim 允许：`fact | hypothesis | conclusion`
- 不存在 `deprecated` status：
  - “不适用/需前提”写为 `conditions/limitations`
  - “确证为错”直接删除 claim（审计由 `mem_edit_log`/delta/rollback 保留）

3) **“验证后”的语义是移动**
- hypothesis→conclusion：更新同一 claim 的 `status`，不复制新增。

4) **每个 item 的 claims 数量硬上限：<= 10**
- 超过则判无效（允许一次格式修复重写；仍失败则丢弃该提案）。

5) **RB content 内禁止出现 KB alias（如 `[C12]`）**
- `[C*]` 为 run 内局部 alias，不跨 run 稳定；写入长期记忆会污染。

6) **RB content 禁止“下一步实验指令”**
- RB 仅记录事实/归因/约束，不作为“下一步实验规划器”。

7) **约束默认负约束，但允许“例外正约束”**
- 默认：只允许 `avoid[...]` 这类负约束；
- 若出现正约束（must/prefer），必须显式标记为例外：
  - `allow_positive=true` + `exception_reason=...`

8) **Chroma 中必须区分 doc_type**
- `doc_type=item` 为真值；
- `doc_type=claim` 为派生索引；
- 任意 item 变更（add/update/archive/unarchive/rollback）必须触发 claim docs 重建。

9) **搜索 API 不支持深分页**
- `/api/v1/memories?query=` 明确不支持 cursor/offset 深翻页，避免越翻越慢。

10) **RB learn 的 update 只能修改候选集**
- extractor 输出的 `update.mem_id` 必须属于本次 candidate memories，否则拒绝。

---

## 3) RBMEM_CLAIMS_V1 协议（存入 MemoryItem.content）

### 3.1 顶层结构（key=value block）

- 第一行必须是：`RBMEM_CLAIMS_V1`
- 后续为 `KEY=VALUE` 行，其中 JSON 以一行承载（不可换行；必要时应做 JSON minify）。

建议顶层字段（可扩展）：
- `TOPIC=<string>`：经验主题（短句）
- `SCOPE=<string>`：可选（global/orchestrator/mof_expert/tio2_expert 等）
- `CLAIMS_JSON=<json array>`：claims 列表（核心）

### 3.2 claim 对象字段（CLAIMS_JSON 内）

每个 claim 至少包含：
- `claim_id`：稳定 id（例如 `c1`、`c2`；或 uuid 短串）
- `status`：`fact|hypothesis|conclusion`
- `facts`：**系统注入**（FACTS_JSON 逻辑，结构见 4.2）
- `inference`：模型归因/推断（允许不确定）
- `constraint`：约束（默认负约束；例外正约束需标记）
- `conditions`：适用条件（可空数组）
- `limitations`：限制/不确定性（可空数组）
- `support`：{`count`, `run_ids`...}
- `contra`：{`count`, `run_ids`...}

说明：
- `facts` 必须包含 `source_run_ids`（最少本次 run_id），用于可追溯与后续状态机。
- `inference` 可以包含概率/置信度表达，但不做强校验（避免过度约束）。

---

## 4) 存储与索引（Chroma item/claim 双文档）

### 4.1 item doc（真值）

- `id = mem_id`
- `metadata.doc_type = "item"`
- `document = MemoryItem.content`（RBMEM_CLAIMS_V1 全文）

### 4.2 claim doc（派生索引，仅用于检索）

- `id = {mem_id}::claim::{claim_id}`
- `metadata.doc_type = "claim"`
- `metadata.parent_mem_id = mem_id`
- `metadata.claim_id = claim_id`
- （可选）`metadata.claim_status = fact|hypothesis|conclusion`
- `document = claim_text_projection`：
  - 用于 embedding 的短文本（例如 claim 的一句话核心 + conditions 的短串）
  - 不强求包含全部 facts/inference/constraint（避免泄露过多噪声进 embedding）

### 4.3 强一致策略

任何 item 变更都必须：
- delete old claim docs（按 parent_mem_id 或 id 前缀）
- parse item.content → claims（<=10）
- rebuild claim docs

提供脚本：
- `rebuild_claim_index`：全库修复 claim docs（运维兜底）

---

## 5) 检索统一语义（claim-search → item-return）

### 5.1 Agent 工具：mem_search

- 只检索 `doc_type=claim`
- 聚合到 item：按 `parent_mem_id` 去重并计算 item 分数（min distance / best match）
- 返回：
  - item（真值）
  - `matched_claims`（claim_id/snippet/distance）写入 trace（提升可解释性）

### 5.2 WebUI 搜索：/api/v1/memories?query=

- 与 mem_search 共用同一套 claim-search 聚合逻辑
- 明确：这是语义检索（embedding）
- 不支持深分页（400），鼓励 refine query 或增大 limit

### 5.3 Browse 列表：/api/v1/memories

- 继续走 SQLite `rb_mem_index` newest-first 分页
- browse/详情页只展示 item，不暴露 claim docs

---

## 6) RB learn 升级（结构化 + facts 注入 + 候选集 + 状态机）

### 6.1 FACTS 系统注入（减少幻觉、提升复盘）

FACTS 由代码从 DB/trace 拼装并注入 extractor 的上下文，并最终写入 `claim.facts`，至少包含：
- `run_id`
- `run_output`：M1/M2/atomic_ratio/modifier 原文（结构化字段 + 原文片段）
- `feedback_products`：topN 产物的 `{name,value,fraction}`
- `activity_total_value`：sum(value)（用于区分活性 vs 选择性）
- `feedback_text`：pros/cons/other（可截断）

### 6.2 质量门禁（写入前）

对每条 add/update 提案：
- 必须能 parse 为 RBMEM_CLAIMS_V1
- claims<=10
- 不含 `[C*]`
- 不含 next-step 实验指令
- constraint 合规（默认负约束；正约束必须显式例外标记）

失败处理：
- 允许 1 次“格式修复重写”（format-only rewrite）
- 仍失败：丢弃该提案（避免污染库）

### 6.3 candidate memories（控制 update 范围）

候选集定义：
- RB learn 本次“允许被 update 的存量 item”集合

选择策略（并集 + 去重 + cap）：
1) 本次 run 实际用过/引用过的 mem（final_output.memory_ids / mem_get / mem_search registry）
2) 用本次事实摘要做 claim-search topK 聚合得到的相关 item
3) 总量 cap（例如 20~30，可配置）

### 6.4 claim 级状态机（support/contra/irrelevant）

每次 learn 后，对候选集 items 的 claims 做三分类，并更新：
- `support.count/run_ids`
- `contra.count/run_ids`

晋升/退化规则（通用且保守）：
- hypothesis → conclusion：support>=2 且 contra==0
- conclusion 遇 contra>=2：至少降回 hypothesis，并要求补充 conditions/limitations；若“明确为错”，则删除 claim

---

## 7) PubChem + COOH 审查（方案 2：展示证据，默认不阻塞）

后端：
- 对 `small_molecule_modifier` 查询 PubChem，尽力解析：
  - CID、Canonical SMILES、InChIKey（可选）
- 基于 SMILES 做 COOH 子结构判定：`has_cooh=true|false`
- 结果缓存（避免频繁请求）
- 查不到/歧义：标记为 unresolved（不自动否决）

前端：
- 在 Run Output 中展示：
  - modifier → CID/SMILES/has_cooh
- `has_cooh=false` 明确提示用户“建议打回/重生成”（但默认不强制阻塞 run）

---

## 8) 迁移策略（本机选 B）

本机（开发/验证）：
- 选 **B：迁移重写** —— 将旧的自由文本 memory 重写为 RBMEM_CLAIMS_V1，并建立 claim 索引。
- 以脚本形式提供（可选调用 LLM；也可先在本机清空 RB 数据重跑）。

服务器（测试版）：
- 直接删数据即可，不做线上迁移。

---

## 9) 验收（DoD）

- `python -m compileall -q src` 通过
- `pytest -q` 通过
- 一次 dry-run batch 可跑通（用于回归 worker/job/DB）
- 实际 run + feedback + RB learn：
  - memory.content 为 RBMEM_CLAIMS_V1，可解析
  - mem_search 命中 claim，返回 item，trace 里有 matched_claims
  - /memories browse 不卡；/memories?query 语义检索生效；不支持深分页
  - PubChem/COOH 信息在 UI 可见（不阻塞 run）

