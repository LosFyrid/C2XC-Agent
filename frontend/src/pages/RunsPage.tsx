import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError } from '../api/client'
import { cancelBatch, createBatch, hideBatch, listBatches, listRuns } from '../api/c2xc'
import type { CreateBatchRequest, RunListItem } from '../api/types'
import { useConfirmDialog } from '../components/ConfirmDialog'
import { DependencyUnavailablePanel } from '../components/DependencyUnavailablePanel'
import { useT } from '../i18n/i18n'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasAnyActiveStatus(value: unknown): boolean {
  if (!isRecord(value)) return false
  const pages = value['pages']
  if (!Array.isArray(pages)) return false
  return pages.some((p) => {
    if (!isRecord(p)) return false
    const items = p['items']
    if (!Array.isArray(items)) return false
    return items.some((r) => {
      if (!isRecord(r)) return false
      const status = r['status']
      return typeof status === 'string' && ['queued', 'running'].includes(status)
    })
  })
}

function formatTs(ts: number | null): string {
  if (!ts) return '—'
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return String(ts)
  }
}

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

function RunsInline(props: { batchId: string }) {
  const t = useT()
  const runsQuery = useInfiniteQuery({
    queryKey: ['runs', props.batchId],
    queryFn: ({ pageParam }) =>
      listRuns({
        batch_id: props.batchId,
        limit: 25,
        cursor: (pageParam as string | undefined) ?? null,
      }),
    enabled: !!props.batchId,
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: (query) => {
      return hasAnyActiveStatus(query.state.data) ? 2000 : false
    },
  })

  if (runsQuery.isLoading) return <div className="text-sm text-muted">{t('common.loading')}</div>
  if (runsQuery.error)
    return (
      <div className="text-sm text-danger">
        {t('common.error')}: {(runsQuery.error as Error).message}
      </div>
    )

  const runs = (runsQuery.data?.pages ?? []).flatMap((p) => p.items ?? [])
  if (runs.length === 0) return <div className="text-sm text-muted">—</div>

  return (
    <div className="mt-2 grid gap-2">
      {runs.map((r: RunListItem) => (
        <div
          key={r.run_id}
          className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface px-3 py-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={r.status} />
            <div className="text-sm font-mono text-fg">{r.run_id}</div>
            <div className="text-xs text-muted">
              #{r.run_index} · {formatTs(r.created_at)}
            </div>
          </div>
          <Link
            to={`/runs/${r.run_id}`}
            className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
          >
            {t('runs.openRun')}
          </Link>
        </div>
      ))}

      <button
        type="button"
        disabled={!runsQuery.hasNextPage || runsQuery.isFetchingNextPage}
        onClick={() => runsQuery.fetchNextPage()}
        className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-fg hover:border-accent disabled:opacity-50"
      >
        {runsQuery.isFetchingNextPage ? t('common.loading') : t('common.loadMore')}
      </button>
    </div>
  )
}

export function RunsPage() {
  const t = useT()
  const queryClient = useQueryClient()
  const { confirm, dialog } = useConfirmDialog()

  const [userRequest, setUserRequest] = useState('')
  const [nRuns, setNRuns] = useState(1)
  const [recipesPerRun, setRecipesPerRun] = useState(1)
  const [temperature, setTemperature] = useState(0.7)
  const [dryRun, setDryRun] = useState(true)

  const [expandedBatchIds, setExpandedBatchIds] = useState<Record<string, boolean>>({})

  const batchesQuery = useInfiniteQuery({
    queryKey: ['batches'],
    queryFn: ({ pageParam }) =>
      listBatches({ limit: 20, cursor: (pageParam as string | undefined) ?? null }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: (query) => {
      // Best-effort polling: keep refreshing while any batch is queued/running.
      return hasAnyActiveStatus(query.state.data) ? 2000 : false
    },
  })

  const batches = useMemo(() => {
    return (batchesQuery.data?.pages ?? []).flatMap((p) => p.items ?? [])
  }, [batchesQuery.data?.pages])

  const createMutation = useMutation({
    mutationFn: (body: CreateBatchRequest) => createBatch(body),
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: ['batches'] })
      if (resp.batch?.batch_id) {
        setExpandedBatchIds((m) => ({ ...m, [resp.batch.batch_id]: true }))
      }
    },
  })

  const [cancelingBatchId, setCancelingBatchId] = useState<string | null>(null)
  const cancelBatchMutation = useMutation({
    mutationFn: (batchId: string) => cancelBatch(batchId),
    onMutate: (batchId) => {
      setCancelingBatchId(batchId)
    },
    onSettled: () => {
      setCancelingBatchId(null)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['batches'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const [hidingBatchId, setHidingBatchId] = useState<string | null>(null)
  const hideBatchMutation = useMutation({
    mutationFn: (batchId: string) => hideBatch(batchId),
    onMutate: (batchId) => {
      setHidingBatchId(batchId)
    },
    onSettled: () => {
      setHidingBatchId(null)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['batches'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const createError = createMutation.error as ApiError | undefined
  const cancelError = cancelBatchMutation.error as ApiError | undefined
  const hideError = hideBatchMutation.error as ApiError | undefined

  const canSubmit = useMemo(() => {
    if (createMutation.isPending) return false
    if (nRuns < 1 || nRuns > 5) return false
    if (recipesPerRun < 1 || recipesPerRun > 3) return false
    return true
  }, [createMutation.isPending, nRuns, recipesPerRun])

  return (
    <div className="grid gap-6">
      <section className="rounded-lg border border-border bg-surface p-4">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-base font-semibold">{t('runs.newBatch')}</h1>
          <button
            type="button"
            onClick={() => batchesQuery.refetch()}
            className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
          >
            {t('runs.refresh')}
          </button>
        </div>

        <form
          className="grid gap-3"
          onSubmit={async (e) => {
            e.preventDefault()
            if (!canSubmit) return

            const details = JSON.stringify(
              {
                user_request: userRequest || '(default)',
                n_runs: nRuns,
                recipes_per_run: recipesPerRun,
                temperature,
                dry_run: dryRun,
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
              intent: dryRun ? 'primary' : 'danger',
            })
            if (!ok) return

            createMutation.mutate({
              user_request: userRequest,
              n_runs: nRuns,
              recipes_per_run: recipesPerRun,
              temperature,
              dry_run: dryRun,
            })
          }}
        >
          <label className="grid gap-1">
            <div className="text-xs text-muted">{t('runs.userRequest')}</div>
            <textarea
              className="min-h-20 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-fg"
              placeholder="(optional) Provide extra constraints, goals, or context…"
              value={userRequest}
              onChange={(e) => setUserRequest(e.target.value)}
            />
          </label>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <label className="grid gap-1">
              <div className="text-xs text-muted">{t('runs.nRuns')}</div>
              <select
                className="rounded-md border border-border bg-bg px-2 py-2 text-sm"
                value={nRuns}
                onChange={(e) => setNRuns(Number(e.target.value))}
              >
                {[1, 2, 3, 4, 5].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-1">
              <div className="text-xs text-muted">{t('runs.recipesPerRun')}</div>
              <select
                className="rounded-md border border-border bg-bg px-2 py-2 text-sm"
                value={recipesPerRun}
                onChange={(e) => setRecipesPerRun(Number(e.target.value))}
              >
                {[1, 2, 3].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-1">
              <div className="text-xs text-muted">{t('runs.temperature')}</div>
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                className="rounded-md border border-border bg-bg px-2 py-2 text-sm"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>

            <label className="flex items-center gap-2 rounded-md border border-border bg-bg px-3 py-2 text-sm">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
              />
              <span className="text-sm">{t('runs.dryRun')}</span>
            </label>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="submit"
              disabled={!canSubmit}
              className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-accent-fg disabled:opacity-50"
            >
              {createMutation.isPending ? t('common.loading') : t('runs.create')}
            </button>
            {createError && createError.code !== 'dependency_unavailable' ? (
              <div className="text-sm text-danger">
                {createError.code}: {createError.message}
              </div>
            ) : null}
          </div>

          {createError && createError.code === 'dependency_unavailable' ? (
            <DependencyUnavailablePanel error={createError} />
          ) : null}
        </form>
      </section>

      <section className="grid gap-3">
        {batchesQuery.isLoading ? (
          <div className="text-sm text-muted">{t('common.loading')}</div>
        ) : batchesQuery.error ? (
          <div className="text-sm text-danger">
            {t('common.error')}: {(batchesQuery.error as Error).message}
          </div>
        ) : batches.length === 0 ? (
          <div className="text-sm text-muted">{t('runs.noBatches')}</div>
        ) : (
          batches.map((b) => {
            const expanded = !!expandedBatchIds[b.batch_id]
            const canCancel = b.status === 'queued' || b.status === 'running'
            const cancelingThis = cancelingBatchId === b.batch_id && cancelBatchMutation.isPending
            const hidingThis = hidingBatchId === b.batch_id && hideBatchMutation.isPending
            return (
              <div key={b.batch_id} className="rounded-lg border border-border bg-surface p-4">
                <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
                  <div className="grid gap-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusBadge status={b.status} />
                      <div className="font-mono text-sm text-fg">{b.batch_id}</div>
                      <div className="text-xs text-muted">{formatTs(b.created_at)}</div>
                    </div>
                    <div className="text-sm text-fg">{b.user_request}</div>
                    <div className="text-xs text-muted">
                      n_runs={b.n_runs} · recipes_per_run={b.recipes_per_run} · started=
                      {formatTs(b.started_at)} · ended={formatTs(b.ended_at)}
                    </div>
                    {b.error ? <div className="text-xs text-danger">{b.error}</div> : null}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => cancelBatchMutation.mutate(b.batch_id)}
                      disabled={!canCancel || cancelBatchMutation.isPending}
                      className="h-9 rounded-md border border-border bg-bg px-3 text-sm text-fg hover:border-danger disabled:opacity-50"
                      title={!canCancel ? t('common.noopTerminal') : undefined}
                    >
                      {cancelingThis ? t('common.loading') : t('batch.cancel')}
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        if (hideBatchMutation.isPending) return
                        const details = JSON.stringify(
                          {
                            batch_id: b.batch_id,
                            status: b.status,
                            created_at: b.created_at,
                            user_request: b.user_request,
                            n_runs: b.n_runs,
                            recipes_per_run: b.recipes_per_run,
                          },
                          null,
                          2,
                        )
                        const ok = await confirm({
                          title: t('confirm.removeBatch.title'),
                          message: t('confirm.removeBatch.message'),
                          details: <pre className="whitespace-pre-wrap font-mono">{details}</pre>,
                          confirmText: t('confirm.confirm'),
                          cancelText: t('confirm.cancel'),
                          intent: 'danger',
                        })
                        if (!ok) return
                        hideBatchMutation.mutate(b.batch_id)
                      }}
                      disabled={hideBatchMutation.isPending || cancelingThis}
                      className="h-9 rounded-md border border-border bg-bg px-3 text-sm text-fg hover:border-danger disabled:opacity-50"
                    >
                      {hidingThis ? t('common.loading') : t('batch.remove')}
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setExpandedBatchIds((m) => ({ ...m, [b.batch_id]: !expanded }))
                      }
                      className="h-9 rounded-md border border-border bg-bg px-3 text-sm text-fg hover:border-accent"
                    >
                      {expanded ? t('runs.collapse') : t('runs.expand')}
                    </button>
                  </div>
                </div>

                {expanded ? <RunsInline batchId={b.batch_id} /> : null}
              </div>
            )
          })
        )}

        {batches.length ? (
          <button
            type="button"
            disabled={!batchesQuery.hasNextPage || batchesQuery.isFetchingNextPage}
            onClick={() => batchesQuery.fetchNextPage()}
            className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-fg hover:border-accent disabled:opacity-50"
          >
            {batchesQuery.isFetchingNextPage ? t('common.loading') : t('common.loadMore')}
          </button>
        ) : null}

        {cancelError ? (
          <div className="text-sm text-danger">
            {cancelError.code}: {cancelError.message}
          </div>
        ) : null}

        {hideError ? (
          <div className="text-sm text-danger">
            {hideError.code}: {hideError.message}
          </div>
        ) : null}
      </section>
      {dialog}
    </div>
  )
}
