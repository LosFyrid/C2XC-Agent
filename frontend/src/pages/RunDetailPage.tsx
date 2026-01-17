import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import {
  cancelRun,
  createBatch,
  getBatch,
  getRun,
  getRunEvidenceAlias,
  getRunEvent,
  getRunModifierChecks,
  getRunOutput,
  listRunEvents,
  resolveRunModifierChecks,
} from '../api/c2xc'
import type { EventListItem, ModifierCheckItem, RunDetailResponse } from '../api/types'
import { CitationMarkdown } from '../components/CitationMarkdown'
import { FeedbackTab } from '../components/FeedbackTab'
import { ReasoningBankTab } from '../components/ReasoningBankTab'
import { useConfirmDialog } from '../components/ConfirmDialog'
import { DependencyUnavailablePanel } from '../components/DependencyUnavailablePanel'
import { JsonViewer } from '../components/JsonViewer'
import { Markdown } from '../components/Markdown'
import { RichTextViewer } from '../components/RichTextViewer'
import { TextViewer } from '../components/TextViewer'
import { useT } from '../i18n/i18n'

function formatTs(ts: number | null): string {
  if (!ts) return '—'
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return String(ts)
  }
}

function aliasSortKey(alias: string): [number, string] {
  const s = (alias ?? '').trim()
  const m = /^([A-Z]+)(\d+)$/.exec(s)
  if (!m) return [1_000_000_000, s]
  const n = Number(m[2])
  return [Number.isFinite(n) ? n : 1_000_000_000, s]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readStringField(obj: unknown, key: string): string | null {
  if (!isRecord(obj)) return null
  const v = obj[key]
  return typeof v === 'string' ? v : null
}

const TRACE_EVENT_PRESETS = {
  all: null,
  important: [
    'run_started',
    'run_failed',
    'run_canceled',
    'final_output',
    'modifier_checks',
    'rb_learn_queued',
    'rb_job_started',
    'rb_job_completed',
    'rb_job_failed',
    'rb_learn_completed',
    'rb_learn_failed',
    'rb_rollback_started',
    'rb_rollback_completed',
  ],
  llm: ['llm_request', 'llm_response'],
  kb: ['kb_query', 'kb_get', 'kb_list'],
  recap: ['recap_info'],
  rb: [
    'rb_unavailable',
    'rb_learn_queued',
    'rb_learn_snapshot',
    'rb_job_started',
    'rb_job_completed',
    'rb_job_failed',
    'rb_learn_completed',
    'rb_learn_failed',
    'rb_source_opened',
    'rb_rollback_started',
    'rb_rollback_completed',
    'mem_search',
    'mem_get',
    'mem_list',
    'memories_resolved',
  ],
} as const

type TracePreset = keyof typeof TRACE_EVENT_PRESETS

function StatusBadge(props: { status: string }) {
  const status = props.status
  const cls =
    status === 'completed'
      ? 'border-success text-success'
      : status === 'failed'
        ? 'border-danger text-danger'
        : status === 'canceled'
          ? 'border-warn text-warn'
          : status === 'running'
            ? 'border-accent text-accent'
            : 'border-border text-muted'

  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs ${cls}`}>
      {status}
    </span>
  )
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

export function RunDetailPage() {
  const t = useT()
  const params = useParams()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const runId = params.runId ?? ''
  const { confirm, dialog } = useConfirmDialog()

  const [tab, setTab] = useState<'output' | 'evidence' | 'trace' | 'feedback' | 'reasoningbank'>(
    'output',
  )
  const [selectedAlias, setSelectedAlias] = useState<string | null>(null)
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [evidenceFilter, setEvidenceFilter] = useState('')
  const [tracePreset, setTracePreset] = useState<TracePreset>('all')

  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = (query.state.data as RunDetailResponse | undefined)?.run?.status
      return status === 'running' || status === 'queued' ? 1500 : false
    },
  })

  const runStatus = runQuery.data?.run?.status ?? ''

  const batchId = runQuery.data?.run?.batch_id ?? null
  const batchQuery = useQuery({
    queryKey: ['batch', batchId],
    queryFn: () => getBatch(batchId as string),
    enabled: !!batchId,
  })

  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['run', runId] })
      queryClient.invalidateQueries({ queryKey: ['events', runId] })
      queryClient.invalidateQueries({ queryKey: ['batches'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const rerunMutation = useMutation({
    mutationFn: async () => {
      const batch = batchQuery.data?.batch
      const snapshot = (batchQuery.data?.batch?.config_snapshot ?? {}) as Record<string, unknown>
      if (!batch) {
        throw new Error('Batch not loaded yet.')
      }
      const temperatureRaw = snapshot.temperature
      const dryRunRaw = snapshot.dry_run

      const temperature =
        typeof temperatureRaw === 'number' && Number.isFinite(temperatureRaw) ? temperatureRaw : 0.7
      const dry_run = Boolean(dryRunRaw)

      return createBatch({
        user_request: batch.user_request,
        n_runs: 1,
        recipes_per_run: batch.recipes_per_run,
        temperature,
        dry_run,
      })
    },
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: ['batches'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      const newRunId = resp.runs?.[0]?.run_id
      if (newRunId) navigate(`/runs/${newRunId}`)
    },
  })

  const outputQuery = useQuery({
    queryKey: ['output', runId],
    queryFn: () => getRunOutput(runId),
    enabled: !!runId,
    refetchInterval: (query) => {
      if (runStatus === 'running' || runStatus === 'queued') return 1500
      // If output not ready yet, keep trying a little.
      const err = query.state.error as ApiError | null
      if (err?.code === 'not_found') return 1500
      return false
    },
  })

  const resolveModifierChecksMutation = useMutation({
    mutationFn: () => resolveRunModifierChecks(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['modifierChecks', runId] })
    },
  })

  const modifierChecksQuery = useQuery({
    queryKey: ['modifierChecks', runId],
    // Read cached checks only; user explicitly triggers compute via POST.
    queryFn: () => getRunModifierChecks(runId),
    enabled: !!runId && Boolean(outputQuery.data),
    retry: false,
    // Cache in react-query; backend also caches in trace.
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  })

  const citationEntries = useMemo(() => {
    const citations = outputQuery.data?.citations ?? {}
    return Object.entries(citations).sort(([a], [b]) => {
      const ka = aliasSortKey(a)
      const kb = aliasSortKey(b)
      if (ka[0] !== kb[0]) return ka[0] - kb[0]
      return ka[1].localeCompare(kb[1])
    })
  }, [outputQuery.data?.citations])

  const filteredCitationEntries = useMemo(() => {
    const q = evidenceFilter.trim().toLowerCase()
    if (!q) return citationEntries
    return citationEntries.filter(([alias, ref]) => {
      const a = (alias ?? '').toLowerCase()
      const r = (ref ?? '').toLowerCase()
      return a.includes(q) || r.includes(q)
    })
  }, [citationEntries, evidenceFilter])

  const evidenceItemQuery = useQuery({
    queryKey: ['evidenceItem', runId, selectedAlias],
    queryFn: () => getRunEvidenceAlias({ run_id: runId, alias: selectedAlias ?? '' }),
    enabled: !!runId && !!selectedAlias,
  })

  const latestFailureQuery = useQuery({
    queryKey: ['eventsLatest', runId, 'run_failed'],
    queryFn: async () => {
      const page = await listRunEvents({
        run_id: runId,
        limit: 50,
        event_type: ['run_failed'],
        include_payload: true,
      })
      const items = page.items ?? []
      return items.length ? items[items.length - 1] : null
    },
    enabled: !!runId && runStatus === 'failed',
  })

  const latestCanceledQuery = useQuery({
    queryKey: ['eventsLatest', runId, 'run_canceled'],
    queryFn: async () => {
      const page = await listRunEvents({
        run_id: runId,
        limit: 50,
        event_type: ['run_canceled'],
        include_payload: true,
      })
      const items = page.items ?? []
      return items.length ? items[items.length - 1] : null
    },
    enabled: !!runId && runStatus === 'canceled',
  })

  const presetTypes = TRACE_EVENT_PRESETS[tracePreset]
  const traceEventTypes = presetTypes ? [...presetTypes] : undefined

  const eventsQuery = useInfiniteQuery({
    queryKey: ['events', runId, tracePreset],
    queryFn: ({ pageParam }) =>
      listRunEvents({
        run_id: runId,
        limit: 50,
        cursor: (pageParam as string | undefined) ?? null,
        event_type: traceEventTypes,
      }),
    enabled: !!runId,
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: () => {
      return runStatus === 'running' || runStatus === 'queued' ? 1500 : false
    },
  })

  const events = useMemo(() => {
    return (eventsQuery.data?.pages ?? []).flatMap((p) => p.items ?? [])
  }, [eventsQuery.data?.pages])

  const eventDetailQuery = useQuery({
    queryKey: ['event', runId, selectedEventId],
    queryFn: () => getRunEvent({ run_id: runId, event_id: selectedEventId ?? '' }),
    enabled: !!runId && !!selectedEventId,
  })

  if (!runId) {
    return (
      <div className="rounded-md border border-border bg-surface p-4 text-sm text-danger">
        Missing runId.
      </div>
    )
  }

  const canCancelRun = runStatus === 'queued' || runStatus === 'running'
  const rerunDryRun = Boolean((batchQuery.data?.batch?.config_snapshot ?? {})['dry_run'])

  const failureErrorText = readStringField(latestFailureQuery.data?.payload, 'error')
  const failureTracebackText = readStringField(latestFailureQuery.data?.payload, 'traceback')
  const canceledReasonText = readStringField(latestCanceledQuery.data?.payload, 'reason')

  const handleOpenEvidenceAlias = (alias: string) => {
    setSelectedAlias(alias)
    setTab('evidence')
  }

  const handleRerun = async () => {
    if (rerunMutation.isPending) return
    if (!batchId || batchQuery.isLoading || batchQuery.isError) return

    const batch = batchQuery.data?.batch
    const snapshot = (batchQuery.data?.batch?.config_snapshot ?? {}) as Record<string, unknown>
    const temperatureRaw = snapshot.temperature
    const temperature =
      typeof temperatureRaw === 'number' && Number.isFinite(temperatureRaw) ? temperatureRaw : 0.7

    const details = JSON.stringify(
      {
        user_request: batch?.user_request ?? '(unknown)',
        n_runs: 1,
        recipes_per_run: batch?.recipes_per_run ?? '(unknown)',
        temperature,
        dry_run: rerunDryRun,
      },
      null,
      2,
    )

    const ok = await confirm({
      title: t('confirm.agentRun.title'),
      message: t('confirm.agentRun.message'),
      details: <pre className="whitespace-pre-wrap font-mono">{details}</pre>,
      confirmText: t('confirm.confirm'),
      cancelText: t('confirm.cancel'),
      intent: rerunDryRun ? 'primary' : 'danger',
    })
    if (!ok) return

    rerunMutation.mutate()
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Link
            to="/"
            className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-fg hover:border-accent"
          >
            {t('nav.back')}
          </Link>
          <div className="font-mono text-sm">{runId}</div>
          {runQuery.data?.run?.status ? <StatusBadge status={runQuery.data.run.status} /> : null}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRerun}
            disabled={!batchId || batchQuery.isLoading || batchQuery.isError || rerunMutation.isPending}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg hover:border-accent disabled:opacity-50"
            title={!batchId ? t('common.error') : undefined}
          >
            {rerunMutation.isPending ? t('common.loading') : t('run.rerun')}
          </button>
          {canCancelRun ? (
            <button
              type="button"
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg hover:border-danger disabled:opacity-50"
            >
              {t('run.cancel')}
            </button>
          ) : null}
        </div>
      </div>

      {rerunMutation.error ? (
        rerunMutation.error instanceof ApiError &&
        (rerunMutation.error as ApiError).code === 'dependency_unavailable' ? (
          <DependencyUnavailablePanel error={rerunMutation.error as ApiError} />
        ) : (
          <div className="rounded-md border border-danger bg-bg p-3 text-sm text-danger">
            {t('common.error')}: {(rerunMutation.error as Error).message}
          </div>
        )
      ) : null}

      <section className="rounded-lg border border-border bg-surface p-4">
        {runQuery.isLoading ? (
          <div className="text-sm text-muted">{t('common.loading')}</div>
        ) : runQuery.error ? (
          <div className="text-sm text-danger">
            {t('common.error')}: {(runQuery.error as Error).message}
          </div>
        ) : (
          <div className="grid gap-1 text-sm">
            <div className="flex flex-wrap gap-x-6 gap-y-1">
              <div>
                <span className="text-muted">{t('run.status')}:</span> {runQuery.data?.run.status}
              </div>
              <div>
                <span className="text-muted">{t('run.createdAt')}:</span>{' '}
                {formatTs(runQuery.data?.run.created_at ?? null)}
              </div>
              <div>
                <span className="text-muted">{t('run.startedAt')}:</span>{' '}
                {formatTs(runQuery.data?.run.started_at ?? null)}
              </div>
              <div>
                <span className="text-muted">{t('run.endedAt')}:</span>{' '}
                {formatTs(runQuery.data?.run.ended_at ?? null)}
              </div>
            </div>
            {runQuery.data?.run.error ? (
              <div className="mt-2 rounded-md border border-danger bg-bg p-3 text-sm text-danger">
                <div className="font-medium">{t('run.error')}</div>
                <div className="mt-1 whitespace-pre-wrap font-mono text-xs">{runQuery.data.run.error}</div>
              </div>
            ) : null}
          </div>
        )}
      </section>

      {runStatus === 'failed' ? (
        <section className="rounded-lg border border-danger bg-surface p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-medium text-danger">{t('run.failureSummary')}</div>
            {latestFailureQuery.data?.event_id ? (
              <button
                type="button"
                onClick={() => {
                  setSelectedEventId(latestFailureQuery.data?.event_id ?? null)
                  setTab('trace')
                }}
                className="rounded-md border border-danger bg-bg px-2 py-1 text-xs text-danger hover:opacity-90"
              >
                {t('run.openInTrace')}
              </button>
            ) : null}
          </div>

          {latestFailureQuery.isLoading ? (
            <div className="text-sm text-muted">{t('common.loading')}</div>
          ) : latestFailureQuery.error ? (
            <div className="text-sm text-danger">
              {t('common.error')}: {(latestFailureQuery.error as Error).message}
            </div>
          ) : latestFailureQuery.data?.payload ? (
            <div className="grid gap-2">
              {failureErrorText ? (
                <div className="rounded-md border border-border bg-bg p-3 text-xs text-fg">
                  <div className="mb-1 text-[11px] font-medium text-muted">error</div>
                  <div className="whitespace-pre-wrap font-mono">{failureErrorText}</div>
                </div>
              ) : null}
              {failureTracebackText ? (
                <div className="rounded-md border border-border bg-bg p-3">
                  <div className="mb-2 text-xs font-medium text-muted">{t('run.traceback')}</div>
                  <TextViewer text={failureTracebackText} />
                </div>
              ) : null}
            </div>
          ) : (
            <div className="text-sm text-muted">—</div>
          )}
        </section>
      ) : null}

      {runStatus === 'canceled' ? (
        <section className="rounded-lg border border-warn bg-surface p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-medium text-warn">{t('run.canceledSummary')}</div>
            {latestCanceledQuery.data?.event_id ? (
              <button
                type="button"
                onClick={() => {
                  setSelectedEventId(latestCanceledQuery.data?.event_id ?? null)
                  setTab('trace')
                }}
                className="rounded-md border border-warn bg-bg px-2 py-1 text-xs text-warn hover:opacity-90"
              >
                {t('run.openInTrace')}
              </button>
            ) : null}
          </div>

          {latestCanceledQuery.isLoading ? (
            <div className="text-sm text-muted">{t('common.loading')}</div>
          ) : latestCanceledQuery.error ? (
            <div className="text-sm text-danger">
              {t('common.error')}: {(latestCanceledQuery.error as Error).message}
            </div>
          ) : latestCanceledQuery.data?.payload ? (
            <div className="rounded-md border border-border bg-bg p-3 text-xs text-fg">
              <div className="mb-1 text-[11px] font-medium text-muted">reason</div>
              <div className="whitespace-pre-wrap font-mono">
                {canceledReasonText ?? ''}
              </div>
            </div>
          ) : (
            <div className="text-sm text-muted">—</div>
          )}
        </section>
      ) : null}

      <section className="rounded-lg border border-border bg-surface p-4">
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setTab('output')}
            className={`rounded-md border px-3 py-2 text-sm ${
              tab === 'output' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            {t('run.output')}
          </button>
          <button
            type="button"
            onClick={() => setTab('evidence')}
            className={`rounded-md border px-3 py-2 text-sm ${
              tab === 'evidence'
                ? 'border-accent text-accent'
                : 'border-border text-fg hover:border-accent'
            }`}
          >
            {t('run.evidence')}
          </button>
          <button
            type="button"
            onClick={() => setTab('trace')}
            className={`rounded-md border px-3 py-2 text-sm ${
              tab === 'trace' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            {t('run.trace')}
          </button>
          <button
            type="button"
            onClick={() => setTab('feedback')}
            className={`rounded-md border px-3 py-2 text-sm ${
              tab === 'feedback'
                ? 'border-accent text-accent'
                : 'border-border text-fg hover:border-accent'
            }`}
          >
            {t('run.feedback')}
          </button>
          <button
            type="button"
            onClick={() => setTab('reasoningbank')}
            className={`rounded-md border px-3 py-2 text-sm ${
              tab === 'reasoningbank'
                ? 'border-accent text-accent'
                : 'border-border text-fg hover:border-accent'
            }`}
          >
            {t('run.reasoningbank')}
          </button>
        </div>

        {tab === 'output' ? (
          <div className="mt-4 grid gap-3">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted">recipes_json</div>
              <button
                type="button"
                onClick={async () => {
                  const text = JSON.stringify(outputQuery.data?.recipes_json ?? {}, null, 2)
                  const ok = await copyText(text)
                  setCopied(ok)
                  setTimeout(() => setCopied(false), 1200)
                }}
                className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
              >
                {copied ? t('common.copied') : t('common.copy')}
              </button>
            </div>

            {outputQuery.isLoading ? (
              <div className="text-sm text-muted">{t('common.loading')}</div>
            ) : outputQuery.error ? (
              (outputQuery.error as ApiError).code === 'not_found' ? (
                <div className="text-sm text-muted">{t('run.outputNotReady')}</div>
              ) : (
                <div className="text-sm text-danger">
                  {t('common.error')}: {(outputQuery.error as Error).message}
                </div>
              )
            ) : (
              <>
                <StructuredRecipes
                  recipesJson={outputQuery.data?.recipes_json}
                  citationAliases={new Set(Object.keys(outputQuery.data?.citations ?? {}))}
                  knownMemIds={new Set(outputQuery.data?.memory_ids ?? [])}
                  onClickAlias={handleOpenEvidenceAlias}
                  modifierChecks={modifierChecksQuery.data?.items ?? []}
                  modifierChecksError={modifierChecksQuery.error as Error | null}
                  modifierChecksLoading={modifierChecksQuery.isLoading}
                  modifierChecksNotFound={(modifierChecksQuery.error as ApiError | null)?.code === 'not_found'}
                  modifierChecksResolving={resolveModifierChecksMutation.isPending}
                  onResolveModifierChecks={() => resolveModifierChecksMutation.mutate()}
                />

                <CitationsPanel
                  citations={outputQuery.data?.citations ?? {}}
                  onClickAlias={handleOpenEvidenceAlias}
                />

                <div className="rounded-md border border-border bg-bg p-3">
                  <div className="mb-2 text-xs font-medium text-muted">raw_output</div>
                  <JsonViewer value={outputQuery.data} defaultMode="tree" />
                </div>
              </>
            )}
          </div>
        ) : null}

        {tab === 'evidence' ? (
          <div className="mt-4 grid gap-4 md:grid-cols-[320px_1fr]">
            <div className="rounded-md border border-border bg-bg p-3">
              <div className="mb-2 text-xs font-medium text-muted">{t('run.aliases')}</div>

              <div className="mb-2">
                <input
                  value={evidenceFilter}
                  onChange={(e) => setEvidenceFilter(e.target.value)}
                  placeholder={t('common.searchPlaceholder')}
                  className="h-8 w-full rounded-md border border-border bg-bg px-2 text-xs text-fg"
                />
              </div>

              {outputQuery.isLoading ? (
                <div className="text-sm text-muted">{t('common.loading')}</div>
              ) : outputQuery.error ? (
                (outputQuery.error as ApiError).code === 'not_found' ? (
                  <div className="text-sm text-muted">{t('run.evidenceAfterOutput')}</div>
                ) : (
                  <div className="text-sm text-danger">
                    {t('common.error')}: {(outputQuery.error as Error).message}
                  </div>
                )
              ) : citationEntries.length === 0 ? (
                <div className="text-sm text-muted">{t('run.citationsEmpty')}</div>
              ) : filteredCitationEntries.length === 0 ? (
                <div className="text-sm text-muted">{t('common.noMatches')}</div>
              ) : (
                <div className="grid gap-1">
                  {filteredCitationEntries.map(([alias, ref]) => (
                    <button
                      key={alias}
                      type="button"
                      onClick={() => setSelectedAlias(alias)}
                      className={`w-full rounded-md border px-2 py-2 text-left text-xs ${
                        selectedAlias === alias
                          ? 'border-accent text-accent'
                          : 'border-border text-fg hover:border-accent'
                      }`}
                    >
                      <div className="font-mono">{alias}</div>
                      <div className="mt-1 break-words font-mono text-[11px] text-muted">{ref}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-md border border-border bg-bg p-3">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-xs font-medium text-muted">
                  {selectedAlias ? `[${selectedAlias}]` : '—'}
                </div>
              </div>

              {selectedAlias ? (
                evidenceItemQuery.isLoading ? (
                  <div className="text-sm text-muted">{t('common.loading')}</div>
                ) : evidenceItemQuery.error ? (
                  <div className="text-sm text-danger">
                    {t('common.error')}: {(evidenceItemQuery.error as Error).message}
                  </div>
                ) : (
                  <div className="grid gap-2">
                    <div className="grid gap-1 text-xs">
                      <div className="break-words text-muted">
                        <span className="font-medium text-fg">source:</span>{' '}
                        <span className="font-mono">{evidenceItemQuery.data?.source || '—'}</span>
                      </div>
                      <div className="break-words text-muted">
                        <span className="font-medium text-fg">ref:</span>{' '}
                        <span className="font-mono">{evidenceItemQuery.data?.ref || '—'}</span>
                      </div>
                      <div className="break-words text-muted">
                        <span className="font-medium text-fg">kb_namespace:</span>{' '}
                        <span className="font-mono">{evidenceItemQuery.data?.kb_namespace || '—'}</span>
                      </div>
                      <div className="break-words text-muted">
                        <span className="font-medium text-fg">lightrag_chunk_id:</span>{' '}
                        <span className="font-mono">{evidenceItemQuery.data?.lightrag_chunk_id || '—'}</span>
                      </div>
                      <div className="break-words text-muted">
                        <span className="font-medium text-fg">created_at:</span>{' '}
                        <span className="font-mono">
                          {formatTs(evidenceItemQuery.data?.created_at ?? null)}
                        </span>
                      </div>
                    </div>

                    <RichTextViewer text={String(evidenceItemQuery.data?.content ?? '')} />
                  </div>
                )
              ) : (
                <div className="text-sm text-muted">{t('run.selectAlias')}</div>
              )}
            </div>
          </div>
        ) : null}

        {tab === 'trace' ? (
          <div className="mt-4 grid gap-4 md:grid-cols-[360px_1fr]">
            <div className="rounded-md border border-border bg-bg p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs font-medium text-muted">{t('run.events')}</div>
                <div className="flex flex-wrap gap-1">
                  {(['all', 'important', 'llm', 'kb', 'recap', 'rb'] as TracePreset[]).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => {
                        setTracePreset(p)
                        setSelectedEventId(null)
                      }}
                      className={`rounded-md border px-2 py-1 text-[11px] ${
                        tracePreset === p
                          ? 'border-accent text-accent'
                          : 'border-border text-fg hover:border-accent'
                      }`}
                    >
                      {t(`trace.preset.${p}`)}
                    </button>
                  ))}
                </div>
              </div>
              {eventsQuery.isLoading ? (
                <div className="text-sm text-muted">{t('common.loading')}</div>
              ) : eventsQuery.error ? (
                <div className="text-sm text-danger">
                  {t('common.error')}: {(eventsQuery.error as Error).message}
                </div>
              ) : events.length === 0 ? (
                <div className="text-sm text-muted">{t('run.eventsEmpty')}</div>
              ) : (
                <div className="grid gap-1">
                  {events.map((e: EventListItem) => (
                    <button
                      key={e.event_id}
                      type="button"
                      onClick={() => setSelectedEventId(e.event_id)}
                      className={`w-full rounded-md border px-2 py-2 text-left text-xs ${
                        selectedEventId === e.event_id
                          ? 'border-accent text-accent'
                          : 'border-border text-fg hover:border-accent'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-mono">{e.event_type}</div>
                        <div className="text-[10px] text-muted">{formatTs(e.created_at)}</div>
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-muted">{e.event_id}</div>
                    </button>
                  ))}
                </div>
              )}

              <div className="mt-3">
                <button
                  type="button"
                  disabled={!eventsQuery.hasNextPage || eventsQuery.isFetchingNextPage}
                  onClick={() => eventsQuery.fetchNextPage()}
                  className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg hover:border-accent disabled:opacity-50"
                >
                  {eventsQuery.isFetchingNextPage ? t('common.loading') : t('common.loadMore')}
                </button>
              </div>
            </div>

            <div className="rounded-md border border-border bg-bg p-3">
              <div className="mb-2 text-xs font-medium text-muted">
                {selectedEventId ? selectedEventId : '—'}
              </div>

              {selectedEventId ? (
                eventDetailQuery.isLoading ? (
                  <div className="text-sm text-muted">{t('common.loading')}</div>
                ) : eventDetailQuery.error ? (
                  <div className="text-sm text-danger">
                    {t('common.error')}: {(eventDetailQuery.error as Error).message}
                  </div>
                ) : (
                  <JsonViewer value={eventDetailQuery.data} defaultMode="tree" />
                )
              ) : (
                <div className="text-sm text-muted">{t('run.selectEvent')}</div>
              )}
            </div>
          </div>
        ) : null}

        {tab === 'feedback' ? (
          <div className="mt-4">
            <FeedbackTab runId={runId} />
          </div>
        ) : null}

        {tab === 'reasoningbank' ? (
          <div className="mt-4">
            <ReasoningBankTab
              runId={runId}
              onOpenRBTrace={() => {
                setTab('trace')
                setTracePreset('rb')
              }}
            />
          </div>
        ) : null}
      </section>
      {dialog}
    </div>
  )
}

function StructuredRecipes(props: {
  recipesJson: unknown
  citationAliases?: Set<string>
  knownMemIds?: Set<string>
  onClickAlias?: (alias: string) => void
  modifierChecks?: ModifierCheckItem[]
  modifierChecksLoading?: boolean
  modifierChecksError?: Error | null
  modifierChecksNotFound?: boolean
  modifierChecksResolving?: boolean
  onResolveModifierChecks?: () => void
}) {
  const obj = isRecord(props.recipesJson) ? props.recipesJson : {}
  const recipesRaw = obj['recipes']
  const recipes = Array.isArray(recipesRaw) ? recipesRaw : []
  if (!recipes.length) return null

  const notes = readStringField(obj, 'overall_notes')?.trim() ?? ''
  const checks = props.modifierChecks ?? []
  const checksByQuery = useMemo(() => {
    const m = new Map<string, ModifierCheckItem>()
    for (const it of checks) {
      if (it && typeof it.query === 'string') m.set(it.query, it)
    }
    return m
  }, [checks])

  return (
    <div className="grid gap-3">
      <div className="text-xs font-medium text-muted">recipes</div>
      <div className="grid gap-3 md:grid-cols-2">
	        {recipes.map((item: unknown, idx: number) => {
	          const r = isRecord(item) ? item : {}
	          const m1 = readStringField(r, 'M1') ?? ''
	          const m2 = readStringField(r, 'M2') ?? ''
	          const atomicRatio = readStringField(r, 'atomic_ratio') ?? ''
	          const modifier = readStringField(r, 'small_molecule_modifier') ?? ''
	          const implRaw = r['implementation_conditions']
	          const impl = isRecord(implRaw) ? implRaw : null
	          const implEntries = impl
	            ? Object.entries(impl).filter(([, v]) => {
	                return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
	              })
	            : []
	          const evidenceSummaryRaw = r['evidence_summary']
	          const evidenceSummary = Array.isArray(evidenceSummaryRaw)
	            ? (evidenceSummaryRaw
	                .filter((v): v is string => typeof v === 'string')
	                .map((v) => v.trim())
	                .filter(Boolean) as string[])
	            : []
	          const rationale = readStringField(r, 'rationale') ?? ''
	          const check = modifier ? checksByQuery.get(modifier) ?? null : null
	          const showModifierChecksError = Boolean(props.modifierChecksError) && !props.modifierChecksNotFound
	          const showResolveButton = Boolean(props.onResolveModifierChecks) && Boolean(modifier) && props.modifierChecksNotFound

	          return (
	          <div key={idx} className="rounded-md border border-border bg-surface p-3 text-sm">
	            <div className="flex items-center justify-between">
	              <div className="font-medium">
                #{idx + 1} · {m1 || '?'}-{m2 || '?'}
              </div>
            </div>
            <div className="mt-2 grid gap-1 text-xs">
              <div>
                <span className="text-muted">M1:</span> <span className="font-mono">{m1 || '—'}</span>
              </div>
              <div>
                <span className="text-muted">M2:</span> <span className="font-mono">{m2 || '—'}</span>
              </div>
              <div>
                <span className="text-muted">atomic_ratio:</span>{' '}
                <span className="font-mono">{atomicRatio || '—'}</span>
              </div>
	              <div>
	                <span className="text-muted">small_molecule_modifier:</span>{' '}
	                <span className="font-mono">{modifier || '—'}</span>
	              </div>
	              {implEntries.length ? (
	                <div className="mt-1 rounded-md border border-border bg-bg p-2 text-[11px]">
	                  <div className="mb-1 text-muted">implementation_conditions</div>
	                  <div className="grid gap-1">
	                    {implEntries.map(([k, v]) => (
	                      <div key={k}>
	                        <span className="text-muted">{k}:</span>{' '}
	                        <span className="font-mono">{String(v)}</span>
	                      </div>
	                    ))}
	                  </div>
	                </div>
	              ) : null}
	              {modifier ? (
	                <div className="mt-1 rounded-md border border-border bg-bg p-2 text-[11px]">
	                  <div className="mb-1 flex items-center justify-between">
	                    <div className="text-muted">PubChem / COOH check</div>
	                    <div className="flex items-center gap-2">
	                      {props.modifierChecksResolving ? (
	                        <div className="text-muted">resolving…</div>
	                      ) : props.modifierChecksLoading ? (
	                        <div className="text-muted">loading…</div>
	                      ) : showModifierChecksError ? (
	                        <div className="text-danger">error</div>
	                      ) : props.modifierChecksNotFound ? (
	                        <div className="text-muted">not computed</div>
	                      ) : null}
	                      {showResolveButton ? (
	                        <button
	                          type="button"
	                          onClick={() => props.onResolveModifierChecks?.()}
	                          disabled={props.modifierChecksResolving}
	                          className="rounded-md border border-border bg-bg px-2 py-1 text-[11px] text-fg hover:border-accent disabled:opacity-50"
	                        >
	                          Resolve
	                        </button>
	                      ) : null}
	                    </div>
	                  </div>
	                  {check ? (
	                    <div className="grid gap-1">
                      <div>
                        <span className="text-muted">status:</span>{' '}
                        <span className="font-mono">{check.status}</span>
                        {check.status === 'resolved' ? (
                          <>
                            {' · '}
                            <span className="text-muted">CID:</span>{' '}
                            <span className="font-mono">{check.cid ?? '—'}</span>
                          </>
                        ) : null}
                      </div>
                      {check.has_cooh === true ? (
                        <div className="text-success">
                          COOH: <span className="font-mono">true</span>
                        </div>
                      ) : check.has_cooh === false ? (
                        <div className="text-danger">
                          COOH: <span className="font-mono">false</span> (建议人工打回/重生成)
                        </div>
                      ) : (
                        <div className="text-muted">
                          COOH: <span className="font-mono">unknown</span>
                        </div>
                      )}
                      {check.canonical_smiles ? (
                        <div className="break-words">
                          <span className="text-muted">SMILES:</span>{' '}
                          <span className="font-mono">{check.canonical_smiles}</span>
                        </div>
                      ) : null}
                      {check.error ? (
                        <div className="text-muted">
                          <span className="text-muted">note:</span> {check.error}
                        </div>
	                      ) : null}
	                    </div>
	                  ) : props.modifierChecksLoading || props.modifierChecksResolving ? (
	                    <div className="text-muted">Loading modifier checks…</div>
	                  ) : props.modifierChecksNotFound ? (
	                    <div className="text-muted">Not computed yet. Click Resolve.</div>
	                  ) : (
	                    <div className="text-muted">No modifier check result.</div>
	                  )}
	                </div>
	              ) : null}
	              {evidenceSummary.length ? (
	                <div className="mt-1 rounded-md border border-border bg-bg p-2 text-[11px]">
	                  <div className="mb-1 text-muted">evidence_summary</div>
	                  <CitationMarkdown
	                    text={evidenceSummary.map((line) => `- ${line}`).join('\n')}
	                    knownAliases={props.citationAliases}
	                    onClickAlias={props.onClickAlias}
	                    knownMemIds={props.knownMemIds}
	                  />
	                </div>
	              ) : null}
	            </div>
	            {rationale ? (
	              <div className="mt-3 text-xs text-fg">
	                <CitationMarkdown
                  text={rationale}
                  knownAliases={props.citationAliases}
                  onClickAlias={props.onClickAlias}
                  knownMemIds={props.knownMemIds}
                />
              </div>
            ) : null}
          </div>
          )
        })}
      </div>
      {notes ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs text-fg">
          <div className="mb-1 text-[11px] font-medium text-muted">overall_notes</div>
          <div className="max-h-[420px] overflow-auto rounded-md border border-border bg-surface p-3">
            <Markdown text={notes} />
          </div>
        </div>
      ) : null}
    </div>
  )
}

function CitationsPanel(props: {
  citations: Record<string, string>
  onClickAlias?: (alias: string) => void
}) {
  const t = useT()
  const entries = Object.entries(props.citations ?? {}).sort(([a], [b]) => {
    const ka = aliasSortKey(a)
    const kb = aliasSortKey(b)
    if (ka[0] !== kb[0]) return ka[0] - kb[0]
    return ka[1].localeCompare(kb[1])
  })

  return (
    <div className="rounded-md border border-border bg-bg p-3">
      <div className="mb-2 text-xs font-medium text-muted">{t('run.citations')}</div>
      {entries.length === 0 ? (
        <div className="text-sm text-muted">{t('run.citationsEmpty')}</div>
      ) : (
        <div className="grid gap-1">
          {entries.map(([alias, ref]) => (
            <button
              key={alias}
              type="button"
              onClick={() => props.onClickAlias?.(alias)}
              className="w-full rounded-md border border-border bg-surface px-2 py-2 text-left text-xs text-fg hover:border-accent"
              title={t('run.openEvidence')}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="font-mono">[{alias}]</div>
                <div className="text-[10px] text-muted">{t('run.open')}</div>
              </div>
              <div className="mt-1 break-words font-mono text-[11px] text-muted">{ref}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
