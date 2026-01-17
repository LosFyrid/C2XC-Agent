import { apiFetch } from './client'
import type {
  BatchDetailResponse,
  CreateBatchRequest,
  CreateBatchResponse,
  Page,
  BatchListItem,
  RunDetailResponse,
  RunListItem,
  RunOutputResponse,
  ModifierChecksResponse,
  EvidenceItem,
  EventListItem,
  Product,
  ProductPreset,
  RunFeedbackResponse,
  SystemWorkerResponse,
  MemoryItem,
  MemoryResponse,
  RbDeltasResponse,
  RbJobsResponse,
  RbLearnResponse,
  RbRollbackResponse,
} from './types'

export function listBatches(params?: {
  limit?: number
  cursor?: string | null
  status?: string[]
}): Promise<Page<BatchListItem>> {
  return apiFetch('/batches', {
    query: {
      limit: params?.limit ?? 50,
      cursor: params?.cursor ?? undefined,
      status: params?.status,
    },
  })
}

export function getBatch(batchId: string): Promise<BatchDetailResponse> {
  return apiFetch(`/batches/${batchId}`)
}

function _randomIdempotencyKey(): string {
  try {
    if (globalThis.crypto && 'randomUUID' in globalThis.crypto) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (globalThis.crypto as any).randomUUID() as string
    }
  } catch {
    // ignore
  }
  return `c2xc_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

export function createBatch(
  body: CreateBatchRequest,
  opts?: { idempotencyKey?: string },
): Promise<CreateBatchResponse> {
  return apiFetch('/batches', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: {
      'Idempotency-Key': opts?.idempotencyKey ?? _randomIdempotencyKey(),
    },
  })
}

export function listRuns(params?: {
  batch_id?: string
  limit?: number
  cursor?: string | null
  status?: string[]
}): Promise<Page<RunListItem>> {
  return apiFetch('/runs', {
    query: {
      batch_id: params?.batch_id ?? undefined,
      limit: params?.limit ?? 50,
      cursor: params?.cursor ?? undefined,
      status: params?.status,
    },
  })
}

export function getRun(runId: string): Promise<RunDetailResponse> {
  return apiFetch(`/runs/${runId}`)
}

export function cancelRun(runId: string, reason?: string): Promise<{ cancel_id: string; status: string }> {
  return apiFetch(`/runs/${runId}/cancel`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason ?? 'user_cancel' }),
  })
}

export function cancelBatch(
  batchId: string,
  reason?: string,
): Promise<{ cancel_id: string; status: string }> {
  return apiFetch(`/batches/${batchId}/cancel`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason ?? 'user_cancel' }),
  })
}

export function getRunOutput(runId: string): Promise<RunOutputResponse> {
  return apiFetch(`/runs/${runId}/output`)
}

// Read cached modifier checks only (no PubChem requests).
export function getRunModifierChecks(runId: string): Promise<ModifierChecksResponse> {
  return apiFetch(`/runs/${runId}/modifier_checks`)
}

export function resolveRunModifierChecks(runId: string, opts?: { force?: boolean }): Promise<ModifierChecksResponse> {
  return apiFetch(`/runs/${runId}/modifier_checks`, {
    method: 'POST',
    body: JSON.stringify({ force: Boolean(opts?.force) }),
  })
}

export function listRunEvidence(params: {
  run_id: string
  limit?: number
  cursor?: string | null
  include_content?: boolean
}): Promise<Page<EvidenceItem>> {
  return apiFetch(`/runs/${params.run_id}/evidence`, {
    query: {
      limit: params.limit ?? 50,
      cursor: params.cursor ?? undefined,
      include_content: params.include_content ?? false,
    },
  })
}

export function getRunEvidenceAlias(params: { run_id: string; alias: string }): Promise<EvidenceItem> {
  return apiFetch(`/runs/${params.run_id}/evidence/${encodeURIComponent(params.alias)}`)
}

export function listRunEvents(params: {
  run_id: string
  limit?: number
  cursor?: string | null
  event_type?: string[]
  include_payload?: boolean
}): Promise<Page<EventListItem>> {
  return apiFetch(`/runs/${params.run_id}/events`, {
    query: {
      limit: params.limit ?? 50,
      cursor: params.cursor ?? undefined,
      event_type: params.event_type,
      include_payload: params.include_payload ?? false,
    },
  })
}

export function getRunEvent(params: { run_id: string; event_id: string }): Promise<EventListItem> {
  return apiFetch(`/runs/${params.run_id}/events/${params.event_id}`)
}

export function getSystemWorker(): Promise<SystemWorkerResponse> {
  return apiFetch('/system/worker')
}

export function listProducts(params?: {
  limit?: number
  cursor?: string | null
  status?: string[]
}): Promise<Page<Product>> {
  return apiFetch('/products', {
    query: {
      limit: params?.limit ?? 200,
      cursor: params?.cursor ?? undefined,
      status: params?.status,
    },
  })
}

export function createProduct(body: { name: string; status?: string }): Promise<{ product: Product }> {
  return apiFetch('/products', {
    method: 'POST',
    body: JSON.stringify({
      name: body.name,
      status: body.status ?? 'active',
      schema_version: 1,
      extra: {},
    }),
  })
}

export function updateProduct(params: {
  product_id: string
  name?: string | null
  status?: string | null
}): Promise<{ product: Product }> {
  return apiFetch(`/products/${encodeURIComponent(params.product_id)}`, {
    method: 'PUT',
    body: JSON.stringify({
      name: params.name ?? undefined,
      status: params.status ?? undefined,
    }),
  })
}

export function listProductPresets(params?: {
  limit?: number
  cursor?: string | null
  status?: string[]
}): Promise<Page<ProductPreset>> {
  return apiFetch('/product_presets', {
    query: {
      limit: params?.limit ?? 200,
      cursor: params?.cursor ?? undefined,
      status: params?.status,
    },
  })
}

export function createProductPreset(body: {
  name: string
  product_ids: string[]
  status?: string
}): Promise<{ preset: ProductPreset }> {
  return apiFetch('/product_presets', {
    method: 'POST',
    body: JSON.stringify({
      name: body.name,
      product_ids: body.product_ids,
      status: body.status ?? 'active',
      schema_version: 1,
      extra: {},
    }),
  })
}

export function updateProductPreset(params: {
  preset_id: string
  name?: string | null
  product_ids?: string[] | null
  status?: string | null
}): Promise<{ preset: ProductPreset }> {
  return apiFetch(`/product_presets/${encodeURIComponent(params.preset_id)}`, {
    method: 'PUT',
    body: JSON.stringify({
      name: params.name ?? undefined,
      product_ids: params.product_ids ?? undefined,
      status: params.status ?? undefined,
    }),
  })
}

export function getRunFeedback(runId: string): Promise<RunFeedbackResponse> {
  return apiFetch(`/runs/${runId}/feedback`)
}

export function upsertRunFeedback(params: {
  run_id: string
  score: number | null
  pros: string
  cons: string
  other: string
  products: { product_id: string; value: number }[]
}): Promise<RunFeedbackResponse> {
  return apiFetch(`/runs/${params.run_id}/feedback`, {
    method: 'PUT',
    body: JSON.stringify({
      score: params.score,
      pros: params.pros,
      cons: params.cons,
      other: params.other,
      products: params.products,
      schema_version: 1,
      extra: {},
    }),
  })
}

export function listMemories(params?: {
  query?: string
  role?: string[]
  status?: string[]
  type?: string[]
  limit?: number
  cursor?: string | null
}): Promise<Page<MemoryItem>> {
  return apiFetch('/memories', {
    query: {
      query: params?.query?.trim() ? params.query.trim() : undefined,
      role: params?.role,
      status: params?.status,
      type: params?.type,
      limit: params?.limit ?? 50,
      cursor: params?.cursor ?? undefined,
    },
  })
}

export function getMemory(memId: string): Promise<MemoryResponse> {
  return apiFetch(`/memories/${encodeURIComponent(memId)}`)
}

export function createMemory(body: {
  role: string
  content: string
  extra?: Record<string, unknown>
}): Promise<MemoryResponse> {
  return apiFetch('/memories', {
    method: 'POST',
    body: JSON.stringify({
      role: body.role,
      status: 'active',
      type: 'manual_note',
      content: body.content,
      schema_version: 1,
      extra: body.extra ?? {},
    }),
  })
}

export function patchMemory(params: {
  mem_id: string
  status?: string | null
  role?: string | null
  type?: string | null
  content?: string | null
  extra?: Record<string, unknown> | null
}): Promise<MemoryResponse> {
  return apiFetch(`/memories/${encodeURIComponent(params.mem_id)}`, {
    method: 'PATCH',
    body: JSON.stringify({
      status: params.status ?? undefined,
      role: params.role ?? undefined,
      type: params.type ?? undefined,
      content: params.content ?? undefined,
      extra: params.extra ?? undefined,
    }),
  })
}

export function archiveMemory(memId: string, reason?: string): Promise<MemoryResponse> {
  return apiFetch(`/memories/${encodeURIComponent(memId)}/archive`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason ?? 'archive' }),
  })
}

export function learnReasoningBank(runId: string): Promise<RbLearnResponse> {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/reasoningbank/learn`, { method: 'POST' })
}

export function listReasoningBankDeltas(runId: string): Promise<RbDeltasResponse> {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/reasoningbank/deltas`)
}

export function listReasoningBankJobs(runId: string, limit?: number): Promise<RbJobsResponse> {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/reasoningbank/jobs`, {
    query: { limit: limit ?? 20 },
  })
}

export function rollbackReasoningBank(params: {
  run_id: string
  delta_id?: string | null
  reason?: string | null
}): Promise<RbRollbackResponse> {
  return apiFetch(`/runs/${encodeURIComponent(params.run_id)}/reasoningbank/rollback`, {
    method: 'POST',
    body: JSON.stringify({
      delta_id: params.delta_id ?? undefined,
      reason: params.reason ?? undefined,
    }),
  })
}
