# C2XC-Agent API 契约（Draft）

最后更新：2025-12-27

> 本文档用于定义 **WebUI ↔ Backend** 的接口契约（contract），以支持后续按模块推进：WebUI → Feedback → ReasoningBank →（K8s TBD）。  
> 它不是替代 `docs/PROJECT_SPEC.md` 的新“单一事实源”，而是把 spec 里的交互与数据流落成 **可演进的工程契约**。若与 spec 冲突，以 spec 为准。

---

## 0. 设计原则（确保可扩展与鲁棒）

1) **版本化**：所有 HTTP API 走前缀 `/api/v1`。破坏性变更只能通过 `/api/v2` 引入。  
2) **向后兼容优先**：新增字段允许；重命名/删除字段不允许（v1 内）。  
3) **扩展槽**：所有核心资源都预留 `schema_version` + `extra`（JSON object）用于扩展，避免频繁迁移接口。  
4) **分页与按需加载**：trace/events 体积会增长，必须提供分页与 payload 延迟加载能力。  
5) **单实例假设但不偷懒**：默认单机单人、单实例；但接口语义仍需清晰（幂等、取消、错误码、状态机）。  
6) **证据与可追溯**：所有引用（KB chunk alias / mem:<id>）必须可在 UI 中点开查看原文/来源/溯源。  

---

## 1. 通用约定（所有接口适用）

### 1.1 Base URL

- Base：`/api/v1`
- 健康检查：`GET /healthz`（不带 `/api/v1` 也可以，但建议统一走 `/api/v1/healthz`）

### 1.2 身份认证（当前最小实现）

- 默认：本地单机单人使用场景，可先不做 auth。
- 预留：后续可以加 `Authorization: Bearer <token>`，但 v1 响应结构不应依赖 auth 机制。

### 1.3 时间与 ID

- **时间戳**：`created_at/started_at/ended_at` 使用 Unix epoch seconds（float 或 int，精度由服务端决定）。  
- **ID 约定**（字符串）：
  - `batch_id`：`batch_<hex>`（现有 SQLite 实现已使用该前缀）
  - `run_id`：`run_<hex>`
  - `event_id`：`evt_<hex>`
  - `cancel_id`：`cancel_<hex>`
  - `product_id`：UUID（无前缀；spec 倾向 UUID）
  - `preset_id`：UUID（无前缀）
  - `mem_id`：UUID（Chroma document id，通常无前缀）

> 客户端必须把 ID 当作 **不透明字符串**，不要解析其内部结构。

### 1.4 统一错误返回（Error Envelope）

所有非 2xx 响应应返回：

```json
{
  "error": {
    "code": "string",
    "message": "human readable message",
    "details": { "any": "json" }
  }
}
```

建议错误码（可扩展）：

- `invalid_argument`（400）
- `not_found`（404）
- `conflict`（409）：违反唯一约束（如 run 已有 feedback）
- `rate_limited`（429）：后续可选
- `internal`（500）
- `dependency_unavailable`（503）：如 KB/LLM 未配置

### 1.5 幂等（Idempotency）

对“创建类请求”（如 `POST /batches`、`PUT /runs/{run_id}/feedback`）建议支持：

- Header：`Idempotency-Key: <uuid>`（可选）
- 语义：同一个 key + 同样的请求体，多次请求返回同一结果，避免 UI 重试造成重复 batch。

### 1.6 分页（Cursor-based）

列表接口统一支持：

- Query：`limit`（默认 50，最大值由服务端限制）
- Query：`cursor`（opaque string，由服务端返回并解释）

统一响应：

```json
{
  "items": [],
  "next_cursor": "opaque-or-null",
  "has_more": true
}
```

> 兼容性要求：新增字段不破坏旧客户端；旧客户端可忽略未知字段。

---

## 2. Batch / Run / Trace（推荐与调试主链路）

### 2.1 Batch 状态机（建议枚举）

- `queued`：已创建，等待执行（或尚未启动）
- `running`：正在执行（可能包含多个 run）
- `completed`：全部 run 成功完成
- `failed`：至少一个 run 失败且未被取消
- `canceled`：被用户取消（best-effort）

### 2.2 Run 状态机（建议枚举）

- `queued`
- `running`
- `completed`
- `failed`
- `canceled`

### 2.3 创建 batch（触发 n 个 run）

`POST /batches`

Request（v1）：

```json
{
  "user_request": "string",
  "n_runs": 2,
  "recipes_per_run": 3,
  "temperature": 0.7,
  "dry_run": false,
  "overrides": {
    "kb_principles_dir": "",
    "kb_modulation_dir": "",
    "llm_model": "",
    "openai_api_base": ""
  },
  "schema_version": 1,
  "extra": {}
}
```

- `overrides`：仅用于调试/开发期临时覆盖；服务端应记录到 `config_snapshot`，但不建议长期暴露太多内部参数。

Response：

```json
{
  "batch": {
    "batch_id": "batch_xxx",
    "created_at": 0,
    "started_at": null,
    "ended_at": null,
    "status": "queued",
    "user_request": "string",
    "n_runs": 2,
    "recipes_per_run": 3,
    "config_snapshot": {},
    "error": null
  },
  "runs": [
    { "run_id": "run_xxx", "run_index": 1, "status": "queued" }
  ]
}
```

### 2.4 查询 batch / runs

- `GET /batches?limit=&cursor=&status=&q=`：列表（`q` 可对 `user_request` 模糊匹配，服务端可选实现）
- `GET /batches/{batch_id}`：详情
- `GET /runs?batch_id=&limit=&cursor=&status=`：run 列表
- `GET /runs/{run_id}`：run 详情

### 2.5 取消 batch/run（best-effort）

- `POST /batches/{batch_id}/cancel`
- `POST /runs/{run_id}/cancel`

Request：

```json
{ "reason": "user_cancel" }
```

Response：

```json
{ "cancel_id": "cancel_xxx", "status": "requested" }
```

语义约束：

- 取消是 **异步** 的：请求成功 ≠ 立即停止；服务端应尽快在下一次可中断点停止后续步骤。
- 取消必须被 trace：至少要写 `run_canceled`（或 `run_failed` 携带 `cancelled=true`）。

### 2.6 读取 run 输出

- `GET /runs/{run_id}/output`

Response：

```json
{
  "recipes_json": {
    "recipes": [
      {
        "M1": "Cu",
        "M2": "Mo",
        "atomic_ratio": "1:1",
        "small_molecule_modifier": "benzoic acid (-COOH)",
        "rationale": "… [C12]"
      }
    ],
    "overall_notes": "string"
  },
  "citations": {
    "C12": "kb:kb_modulation__chunk-abc"
  },
  "memory_ids": [
    "123e4567-e89b-12d3-a456-426614174000"
  ]
}
```

### 2.6.1 解析 small_molecule_modifier（PubChem / COOH 审查）

> 目的：为实验人员提供 **可复盘的结构证据**（CID/SMILES），并做 best-effort 的 COOH 结构信号提示。  
> 注意：这是 **WebUI 展示/人工审查** 用，不应阻塞 run 本身。

- `POST /runs/{run_id}/modifier_checks`

Request：

```json
{ "force": false }
```

Response：

```json
{
  "run_id": "run_xxx",
  "items": [
    {
      "query": "benzoic acid (-COOH)",
      "normalized_query": "benzoic acid",
      "status": "resolved",
      "cid": 243,
      "canonical_smiles": "O=C(O)c1ccccc1",
      "inchikey": "WPYMKLBDIGXBTP-UHFFFAOYSA-N",
      "has_cooh": true,
      "error": null
    }
  ]
}
```

语义：
- 服务端会将结果 best-effort 缓存为 trace event：`modifier_checks`（避免重复请求外部 PubChem）。
- `has_cooh` 为启发式判定（无 RDKit 时的近似），用于提示；最终以人工核查为准。

### 2.7 读取 trace/events（UI 调试核心）

- `GET /runs/{run_id}/events?limit=&cursor=&event_type=&since=&until=&include_payload=`

参数建议：

- `event_type` 支持多值：`event_type=kb_query&event_type=recap_info`
- `include_payload=false` 时只返回元数据（用于大列表）；点击某条事件再 `GET /runs/{run_id}/events/{event_id}` 拉详情。

事件记录（推荐返回结构）：

```json
{
  "event_id": "evt_xxx",
  "run_id": "run_xxx",
  "created_at": 0,
  "event_type": "kb_query",
  "payload": {}
}
```

#### 2.7.1 事件类型（建议最小集合 + 可扩展）

以下事件类型已在现有代码中出现或可以合理扩展（payload 仅定义关键字段；允许新增字段）：

- `run_started`
  - `mode`：`dry_run | normal`
  - `user_request`, `recipes_per_run`, `temperature`
- `recap_info`
  - `agent`：`orchestrator|mof_expert|tio2_expert`
  - `recap_state`：`down|action_taken|up`
  - `task_name`, `think`, `subtasks`, `result`, `depth`, `steps`
- `kb_query`
  - `kb_namespace`：`kb_principles|kb_modulation`
  - `query`, `mode`, `top_k`
  - `results[]`：`{ alias, ref, source, content, kb_namespace, lightrag_chunk_id }`
- `kb_get`
  - `alias`, `ref`, `source`, `kb_namespace`, `lightrag_chunk_id`
  - `context`：可选（如 `generate_recipes`）
- `kb_list`
  - `total`, `limit`, `shown_aliases[]`
- `llm_request`
  - `model`, `temperature`, `messages[]`（注意体积；可选择只在 include_payload=true 返回）
  - `attempt/turn/steps`, `recap_state`, `task_name`, `agent`
- `llm_response`
  - `content`, `tool_calls[]`, `raw`（注意体积；同上）
- `citations_resolved`
  - `aliases[]`, `resolved{alias->kb_ref}`
- `memories_resolved`
  - `mem_ids[]`（最终输出中出现的 mem_id 列表）
- `final_output`
  - `recipes_json`, `citations`, `memory_ids`
- `mem_search`
  - `query`, `top_k`（可选）
  - `role/status/mem_type`（可选过滤）
  - `results[]`：`{ mem_id, role, type, status, source_run_id }`
- `mem_get`
  - `mem_id`, `role`, `type`, `status`, `source_run_id`
- `mem_list`
  - `total`, `limit`, `shown_mem_ids[]`
- `rb_learn_queued`
  - `rb_job_id`, `kind`, `status`
- `rb_job_started`
  - `rb_job_id`, `kind`
- `rb_job_completed`
  - `rb_job_id`, `kind`, `delta_id`
- `rb_job_failed`
  - `rb_job_id`, `kind`, `error`（可选 `missing[]`）
- `rb_learn_completed`
  - `rb_job_id`, `delta_id`, `n_ops`, `dry_run`
- `rb_learn_failed`
  - `rb_job_id`, `error`, `traceback`
- `rb_learn_snapshot`
  - `rb_job_id`, `snapshot{trace_cutoff_ts,feedback_id,feedback_updated_at,final_output_event_id}`, `budget{...}`, `policy{...}`
- `rb_source_opened`
  - `rb_job_id`, `source_type`, `source_id`, `mode_requested/mode_used`, `returned_chars`, `truncated`, `error_code`（可选）
- `rb_rollback_started`
  - `delta_id`, `reason`, `n_ops`
- `rb_rollback_completed`
  - `delta_id`, `status`
- `run_failed`
  - `error`, `traceback`（可选）
- `run_canceled`
  - `reason`

> 约束：事件 payload 不应包含密钥（如 `OPENAI_API_KEY`）。配置快照可记录模型名/base_url 等非敏感信息。

---

## 3. Evidence（KB 引用别名与证据展示）

UI 需要“按 alias 查看证据 chunk 原文”。v1 推荐两种实现，二选一即可：

### 3.1 方案 A（事件驱动，最少新增表）

- UI 从 `kb_query` 事件中聚合 `results[]`，构建“run evidence registry”。
- 优点：不需要额外表；与 trace 一致。
- 缺点：事件量大时聚合成本更高；分页加载要设计好。

### 3.2 方案 B（显式 evidence API，便于 UI）

新增接口：

- `GET /runs/{run_id}/evidence?limit=&cursor=&include_content=false`
- `GET /runs/{run_id}/evidence/{alias}`

Evidence item：

```json
{
  "alias": "C12",
  "ref": "kb:kb_modulation__chunk-abc",
  "source": "10.xxxx/xxxx OR filename.pdf",
  "content": "full chunk text",
  "kb_namespace": "kb_modulation",
  "lightrag_chunk_id": "chunk-abc",
  "created_at": 0
}
```

> 建议：即使实现方案 A，也可在后续演进到方案 B（不破坏 UI，仅提升性能）。

---

## 4. Feedback（实验员输入契约；RB 的稳定上游）

### 4.1 Products（产物目录）

- `GET /products?limit=&cursor=&status=`
- `GET /products/{product_id}`
- `POST /products`
- `PUT /products/{product_id}`

Product：

```json
{
  "product_id": "uuid",
  "created_at": 0,
  "updated_at": 0,
  "name": "C2H4",
  "status": "active",
  "schema_version": 1,
  "extra": {}
}
```

### 4.2 Presets（产物集合）

- `GET /product_presets?limit=&cursor=&status=`
- `GET /product_presets/{preset_id}`
- `POST /product_presets`
- `PUT /product_presets/{preset_id}`

Preset：

```json
{
  "preset_id": "uuid",
  "created_at": 0,
  "updated_at": 0,
  "name": "Default gas products",
  "product_ids": ["uuid1", "uuid2"],
  "schema_version": 1,
  "extra": {}
}
```

### 4.3 Feedback（每个 run 最多 1 条；fraction 必须持久化）

- `GET /runs/{run_id}/feedback`
- `PUT /runs/{run_id}/feedback`（upsert；推荐 PUT 以强调“同一 run 只有一条记录”）

Request：

```json
{
  "score": 7.5,
  "pros": "string",
  "cons": "string",
  "other": "string",
  "products": [
    { "product_id": "uuid", "value": 12.3 },
    { "product_id": "uuid", "value": 4.5 }
  ],
  "schema_version": 1,
  "extra": {}
}
```

Response（服务端计算并持久化 `fraction`）：

```json
{
  "feedback": {
    "feedback_id": "feedback_xxx",
    "run_id": "run_xxx",
    "score": 7.5,
    "pros": "string",
    "cons": "string",
    "other": "string",
    "products": [
      {
        "feedback_product_id": "fbp_xxx",
        "product_id": "uuid",
        "product_name": "C2H4",
        "product_status": "active",
        "value": 12.3,
        "fraction": 0.7321
      }
    ],
    "created_at": 0,
    "updated_at": 0,
    "schema_version": 1,
    "extra": {}
  }
}
```

业务约束：

- `products[].product_id` 必须存在于 `products` 目录（不允许自由文本）。
- `fraction` 由服务端计算：`value_i / sum(value_all_selected_products)`，并必须存储（避免口径漂移）。
- `products[].product_id` 不允许重复。
- `sum(value)=0` 时，所有 `fraction` 存储为 `0`（同时 UI 应提示）。

---

## 5. ReasoningBank（RB）契约（mem:<id> 可引用 + 严格回滚）

> 具体实现依赖 Chroma 与 RB 管线（retrieval → extraction → consolidation）。本节定义 UI/后端接口的稳定形态，确保可演进。

### 5.1 Memory items（浏览/检索/编辑）

- `GET /memories?query=&role=&status=&type=&limit=&cursor=`
- `GET /memories/{mem_id}`
- `POST /memories`（手工新增 manual_note）
- `PATCH /memories/{mem_id}`（编辑）
- `POST /memories/{mem_id}/archive`（软删除：status=archived）

Memory item（建议字段；允许扩展）：

```json
{
  "mem_id": "uuid",
  "status": "active",
  "role": "global|orchestrator|mof_expert|tio2_expert",
  "type": "reasoningbank_item|manual_note",
  "content": "main text",
  "source_run_id": "run_xxx",
  "created_at": 0,
  "updated_at": 0,
  "schema_version": 1,
  "extra": {}
}
```

引用形式：

- 对外展示/trace：`mem:<mem_id>`

### 5.2 RB 学习触发（由 feedback 驱动）

推荐策略：feedback 保存后自动触发 RB 学习（异步），同时提供手动触发接口：

- `POST /runs/{run_id}/reasoningbank/learn`

Response：

```json
{
  "run_id": "run_xxx",
  "job_id": "rbjob_xxx",
  "status": "queued"
}
```

同时提供（用于 UI 观测与排障）：

- `GET /runs/{run_id}/reasoningbank/jobs?limit=20`

Response（示例）：

```json
{
  "run_id": "run_xxx",
  "jobs": [
    {
      "rb_job_id": "rbjob_xxx",
      "run_id": "run_xxx",
      "kind": "learn",
      "created_at": 0,
      "started_at": 0,
      "ended_at": 0,
      "status": "failed|queued|running|completed",
      "error": "string|null",
      "schema_version": 1,
      "extra": {}
    }
  ]
}
```

### 5.3 严格回滚（delta 记录）

为满足 spec 的“回滚重学”语义，RB 学习必须记录 delta，并提供查询与回滚接口：

- `GET /runs/{run_id}/reasoningbank/deltas`
- `POST /runs/{run_id}/reasoningbank/rollback`（按 delta_id 或按最后一次）

delta 结构（建议）：

```json
{
  "delta_id": "rbd_xxx",
  "run_id": "run_xxx",
  "created_at": 0,
  "ops": [
    { "op": "add", "mem_id": "uuid" },
    { "op": "update", "mem_id": "uuid", "before": {}, "after": {} },
    { "op": "archive", "mem_id": "uuid" }
  ],
  "schema_version": 1,
  "extra": {}
}
```

> 回滚语义：**严格回滚**到提炼前状态，即使 UI 曾对这些 mem 条目做过人工编辑，也会被回退覆盖（spec 已确认）。

---

## 6. K8s（占位）

K8s 的具体接口/鉴权/部署拓扑需求后续对接。本契约仅保证：

- 所有运行时参数可通过 env/config 注入；
- 支持健康检查；
- 存储路径/持久化语义明确（SQLite/KB/RB）。

---

## 7. 系统观测（WebUI 调试辅助）

> 本节为“科研工作台”调试体验提供稳定的最小观测面。它不替代 trace/events，而是回答：
> “worker 是否在跑？队列里有多少 queued/running？是否发生过启动 reconcile？”

### 7.1 Worker 状态与队列统计

- `GET /api/v1/system/worker`

Response（示例）：

```json
{
  "ts": 0,
  "worker": { "enabled": true, "running": true, "poll_interval_s": 0.5 },
  "queue": {
    "runs_by_status": { "queued": 1, "running": 0, "completed": 10 },
    "batches_by_status": { "queued": 0, "running": 0, "completed": 5 },
    "rb_jobs_by_status": { "queued": 1, "running": 0, "completed": 10, "failed": 0 }
  },
  "startup": { "reconciled_running_runs": 0 }
}
```

### 7.2 Debug 端点开关

为避免在非开发环境暴露调试信息，所有 `/api/v1/_debug/*` 端点应默认关闭，并可通过：

- `C2XC_ENABLE_DEBUG_ENDPOINTS=1` 显式启用（默认 `0`）。
