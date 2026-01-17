export type ErrorEnvelope = {
  error: {
    code: string
    message: string
    details?: unknown
  }
}

export type Page<TItem> = {
  items: TItem[]
  has_more: boolean
  next_cursor: string | null
}

export type BatchListItem = {
  batch_id: string
  created_at: number
  started_at: number | null
  ended_at: number | null
  user_request: string
  n_runs: number
  recipes_per_run: number
  status: 'queued' | 'running' | 'completed' | 'failed' | 'canceled' | string
  error: string | null
}

export type RunListItem = {
  run_id: string
  batch_id: string
  run_index: number
  created_at: number
  started_at: number | null
  ended_at: number | null
  status: 'queued' | 'running' | 'completed' | 'failed' | 'canceled' | string
  error: string | null
}

export type BatchDetailResponse = {
  batch: BatchListItem & {
    config_snapshot: Record<string, unknown>
  }
}

export type RunDetailResponse = {
  run: RunListItem
}

export type CreateBatchRequest = {
  user_request: string
  n_runs: number
  recipes_per_run: number
  temperature: number
  dry_run: boolean
}

export type CreateBatchResponse = {
  batch: BatchListItem & {
    config_snapshot: Record<string, unknown>
  }
  runs: RunListItem[]
}

export type RunOutputResponse = {
  recipes_json: unknown
  citations: Record<string, string>
  memory_ids: string[]
}

export type ModifierCheckItem = {
  query: string
  normalized_query: string
  status: string
  cid: number | null
  canonical_smiles: string | null
  inchikey: string | null
  has_cooh: boolean | null
  error: string | null
}

export type ModifierChecksResponse = {
  run_id: string
  items: ModifierCheckItem[]
}

export type EventListItem = {
  event_id: string
  run_id: string
  created_at: number
  event_type: string
  payload?: unknown
}

export type EvidenceItem = {
  alias: string
  ref: string
  source: string
  kb_namespace: string
  lightrag_chunk_id: string | null
  created_at: number
  content?: string
}

export type Product = {
  product_id: string
  created_at: number
  updated_at: number
  name: string
  status: 'active' | 'archived' | string
}

export type ProductPreset = {
  preset_id: string
  created_at: number
  updated_at: number
  name: string
  status: 'active' | 'archived' | string
  product_ids: string[]
}

export type RunFeedbackProduct = {
  feedback_product_id: string
  product_id: string
  product_name: string
  product_status: string
  value: number
  fraction: number
}

export type RunFeedback = {
  feedback_id: string
  run_id: string
  created_at: number
  updated_at: number
  score: number | null
  pros: string
  cons: string
  other: string
  schema_version: number
  extra: Record<string, unknown>
  products: RunFeedbackProduct[]
}

export type RunFeedbackResponse = {
  feedback: RunFeedback
}

export type SystemWorkerResponse = {
  ts: number
  worker: {
    enabled: boolean
    running: boolean
    db_path?: string
    poll_interval_s?: number
  }
  queue: {
    runs_by_status: Record<string, number>
    batches_by_status: Record<string, number>
    rb_jobs_by_status?: Record<string, number>
  }
  startup: {
    reconciled_running_runs: number
  }
}

export type MemoryItem = {
  mem_id: string
  status: 'active' | 'archived' | string
  role: 'global' | 'orchestrator' | 'mof_expert' | 'tio2_expert' | string
  type: 'reasoningbank_item' | 'manual_note' | string
  content: string
  source_run_id: string | null
  created_at: number
  updated_at: number
  schema_version: number
  extra: Record<string, unknown>
  distance?: number
}

export type MemoryResponse = {
  memory: MemoryItem
}

export type RbDeltaOp = {
  op: 'add' | 'update' | 'archive' | string
  mem_id: string
  before?: unknown
  after?: unknown
}

export type RbDelta = {
  delta_id: string
  run_id: string
  created_at: number
  status: 'applied' | 'rolled_back' | string
  rolled_back_at: number | null
  rolled_back_reason: string | null
  ops: RbDeltaOp[]
  schema_version: number
  extra: Record<string, unknown>
}

export type RbDeltasResponse = {
  run_id: string
  deltas: RbDelta[]
}

export type RbJob = {
  rb_job_id: string
  run_id: string
  kind: string
  created_at: number
  started_at: number | null
  ended_at: number | null
  status: 'queued' | 'running' | 'completed' | 'failed' | 'canceled' | string
  error: string | null
  schema_version: number
  extra: Record<string, unknown>
}

export type RbJobsResponse = {
  run_id: string
  jobs: RbJob[]
}

export type RbLearnResponse = {
  run_id: string
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | string
}

export type RbRollbackResponse = {
  run_id: string
  delta_id: string
  status: 'rolled_back' | string
}
