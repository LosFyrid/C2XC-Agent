# Milestone 3：ReasoningBank（Chroma + 可引用 + 严格回滚重学）

最后更新：2025-12-29

> 本文档用于实时跟踪 Milestone 3 的实施进度与关键决策。  
> 关联文档：
> - 总计划：`docs/DEVELOPMENT_PLAN.md`
> - Spec（Single Source of Truth）：`docs/PROJECT_SPEC.md`
> - API 契约：`docs/API_CONTRACT.md`（以当前实现为准，向后兼容演进）

---

## 0) 关键决策（冻结）

- 存储分工（Spec 已定）：
  - ReasoningBank 只用 **Chroma (persistent)** 存储全文与索引（单 collection + metadata 过滤）
  - SQLite 只用于：runs/trace/feedback + RB 的 jobs/deltas + mem_edit_log（审计）
- 引用形态：
  - KB chunk：`[C1]`（alias）
  - RB memory：`mem:<uuid>`（例如 `mem:123e4567-e89b-12d3-a456-426614174000`）
- RB 经验条目协议（Milestone 3.1 升级）：
  - **memory 真值单位仍是 item**（UI 浏览/引用单位不变：`mem:<id>`）
  - **检索单位下沉为 claim**：Chroma 同 collection 内区分 `doc_type=item|claim`，对 `doc_type=claim` 做 embedding 检索，再聚合返回 item
  - `reasoningbank_item.content` 升级为严格结构化 `RBMEM_CLAIMS_V1`（可解析/可校验/可渲染）
  - 详见：`docs/MILESTONE_3_RB_CLAIMS_V1_PLAN.md`
- 严格回滚（硬语义）：
  - RB learn 每次必须记录 delta（add/update/archive 的 ops）
  - feedback 更新触发 RB learn 前：必须先回滚该 run 的已应用 delta，再按新 feedback 重学
  - 回滚覆盖人工编辑（即使 UI 手工改过，也会被回退覆盖）
- 默认启用真实 embeddings（科研部署取向）：
  - `config/default.toml` 默认 `reasoningbank.embedding_mode="openai"`
  - embeddings 与 chat 支持 **分开配置**：
    - chat：`OPENAI_API_BASE` + `OPENAI_API_KEY` + `LLM_MODEL`
    - embeddings：`C2XC_EMBEDDING_API_BASE` + `C2XC_EMBEDDING_API_KEY` + `C2XC_EMBEDDING_MODEL`
  - 离线/测试可用 `hash` embedding（仅用于测试闭环；非科研部署）

---

## 1) 范围（In Scope / Out of Scope）

### In Scope（Milestone 3 做）

- RB‑Browse（可浏览/可引用）
  - `/api/v1/memories` 列表/详情/搜索/编辑/归档
  - WebUI：`/memories` + `/memories/:memId` 页面
  - Trace/Output 中的 `mem:<id>` 可点击跳转
- RB‑Learn（由 feedback 驱动，异步队列）
  - feedback 保存后 best-effort enqueue `rb_jobs`（worker 执行）
  - 也提供手动触发接口：`POST /api/v1/runs/{run_id}/reasoningbank/learn`
- RB‑Consolidate（只合并 near-duplicate，冲突默认保留）
  - consolidation 策略版本化：`strategy_version`
- RB‑Rollback（严格回滚）
  - `GET /api/v1/runs/{run_id}/reasoningbank/deltas`
  - `POST /api/v1/runs/{run_id}/reasoningbank/rollback`

### Out of Scope（后续里程碑做）

- Comparison report（历史对比计算与展示，Milestone 4/5 再细化）
- K8s / 多实例 / 鉴权（Milestone 4/5）

---

## 2) 交付清单（实时更新）

- [x] DB schema_version 升级到 5（v4: RB jobs/deltas/mem_edit_log；v5: `rb_mem_index` 用于稳定/高性能 browse 分页）
- [x] Chroma persistent ReasoningBankStore（单 collection + metadata 过滤）
- [x] Backend API：
  - [x] Memories CRUD（manual_note 可新增；patch + archive）
  - [x] RB learn/deltas/rollback
- [x] Worker：
  - [x] RB jobs 与 runs 共用单 worker（无 queued run 时处理 rb_job）
  - [x] `C2XC_RB_LEARN_DRY_RUN=1` 支持无 LLM 提炼（用于离线测试）
- [x] Agent 接入：
  - [x] structured subtasks 增加 `mem_search/mem_get/mem_list`
  - [x] 输出校验：rationale 必须至少含 `[C*]` 或 `mem:<id>`；mem 引用必须来自 run memory registry 且为 active
  - [x] Run output 增加 `memory_ids`（用于 UI 侧高亮/跳转）
- [x] WebUI：
  - [x] Memories 页面 + 详情页（编辑/归档/激活）
  - [x] Run 详情页增加 ReasoningBank tab（learn + deltas + rollback）
  - [x] Trace presets 增加 `RB`
- [x] Tests（pytest）：
  - [x] Memories API CRUD
  - [x] feedback → enqueue rb_job → worker 执行 learn → rollback
  - [x] strict rollback 覆盖人工编辑（update op）
  - [x] ReCAP mem_search + mem:<id> 引用校验（stub LLM）

---

## 3) 如何运行（Milestone 3 当前可用）

### 3.1 Backend

- 安装：
  - `pip install -r requirements-dev.txt`
- 启动：
  - `python scripts/serve.py`

默认：
- Backend：`http://127.0.0.1:8000`

### 3.2 Frontend

- `cd frontend`
- `npm install`
- `npm run dev`

默认：
- WebUI：`http://127.0.0.1:5173`

---

## 4) ReasoningBank 运行参数（常用）

### 4.1 Chroma 持久化目录

- 默认（来自 `config/default.toml`）：`data/chroma`（在 repo 根目录下）
- 覆盖：
  - `export C2XC_RB_CHROMA_DIR=/abs/path/to/data/chroma`

### 4.2 Embedding 模式

- 科研部署（默认；需要 embeddings provider）：  
  - `export C2XC_RB_EMBEDDING_MODE=openai`
  - `export C2XC_EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`（示例：DashScope OpenAI 兼容网关）
  - `export C2XC_EMBEDDING_API_KEY=...`
  - `export C2XC_EMBEDDING_MODEL=text-embedding-v4`（示例：Qwen embedding）
  - 可选：`export C2XC_EMBEDDING_SEND_DIMENSIONS=1`（只有部分 OpenAI-compatible 端点支持 `dimensions` 参数）
  - ⚠️ 如果你之前用 `hash` 模式创建过同名 collection：切到 `openai` 后请清空 `C2XC_RB_CHROMA_DIR` 或改 `C2XC_RB_COLLECTION` 新建 collection（避免 embedding function 不匹配）
- 离线/测试（可选）：  
  - `export C2XC_RB_EMBEDDING_MODE=hash`

### 4.3 RB learn（无 LLM）

RB learn 在无 LLM 时可通过 dry-run 路径运行（用于验证闭环与回滚语义）：

- `export C2XC_RB_LEARN_DRY_RUN=1`

---

## 5) 验收路径（端到端闭环）

1) WebUI 创建 dry-run batch（Runs 页面）
2) Run 详情页填写/保存 feedback（Feedback tab）
   - 保存后会 best-effort 自动 enqueue RB learn job
3) Run 详情页 → ReasoningBank tab
   - 查看 deltas（为空则可手动点 Learn）
   - 可对最新 applied delta 执行 rollback
4) WebUI → Memories
   - 浏览/搜索/打开 mem 详情
   - `mem:<id>` 在 Output/Trace 中可点击跳转

---

## 6) 已知问题 / 下一步

- `lightrag`（KB）与真实 LLM/embedding 的运行环境依赖仍需要用户自行配置（Milestone 4 会统一容器化与启动自检）。
