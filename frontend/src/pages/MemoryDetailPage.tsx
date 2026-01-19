import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import { archiveMemory, getMemory, patchMemory } from '../api/c2xc'
import { JsonViewer } from '../components/JsonViewer'
import { MemoryContentViewer } from '../components/MemoryContentViewer'
import { useT } from '../i18n/i18n'

function formatTs(ts: number | null): string {
  if (!ts) return '—'
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return String(ts)
  }
}

export function MemoryDetailPage() {
  const t = useT()
  const queryClient = useQueryClient()
  const params = useParams()
  const memId = params.memId ?? ''

  const memoryQuery = useQuery({
    queryKey: ['memory', memId],
    queryFn: () => getMemory(memId),
    enabled: !!memId,
  })

  const memory = memoryQuery.data?.memory

  const [editing, setEditing] = useState(false)
  const [content, setContent] = useState('')
  const [role, setRole] = useState('global')
  const [type, setType] = useState('manual_note')

  const canEdit = useMemo(() => Boolean(memory && editing), [memory, editing])

  const patchMutation = useMutation({
    mutationFn: () =>
      patchMemory({
        mem_id: memId,
        content: content.trim(),
        role,
        type,
      }),
    onSuccess: () => {
      setEditing(false)
      queryClient.invalidateQueries({ queryKey: ['memory', memId] })
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
  })

  const activateMutation = useMutation({
    mutationFn: () =>
      patchMemory({
        mem_id: memId,
        status: 'active',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory', memId] })
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
  })

  const archiveMutation = useMutation({
    mutationFn: () => archiveMemory(memId, 'archive'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory', memId] })
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
  })

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Link to="/memories" className="text-xs text-muted hover:text-fg">
            {t('nav.memories')}
          </Link>
          <span className="text-xs text-muted">/</span>
          <div className="truncate font-mono text-xs text-fg">mem:{memId}</div>
        </div>
        <Link to="/" className="text-xs text-muted hover:text-fg">
          {t('nav.runs')}
        </Link>
      </div>

      {memoryQuery.isLoading ? (
        <div className="text-sm text-muted">{t('common.loading')}</div>
      ) : memoryQuery.error ? (
        <div className="text-sm text-danger">
          {t('common.error')}: {(memoryQuery.error as ApiError).message}
        </div>
      ) : !memory ? (
        <div className="text-sm text-muted">—</div>
      ) : (
        <>
          <section className="rounded-lg border border-border bg-surface p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted">
                  {memory.role}
                </span>
                <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted">
                  {memory.type}
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-xs ${
                    memory.status === 'active' ? 'border-success text-success' : 'border-warn text-warn'
                  }`}
                >
                  {memory.status}
                </span>
                {memory.source_run_id ? (
                  <Link
                    to={`/runs/${encodeURIComponent(memory.source_run_id)}`}
                    className="rounded-full border border-border bg-bg px-2 py-0.5 text-xs text-fg hover:border-accent"
                    title="Open source run"
                  >
                    run:{memory.source_run_id}
                  </Link>
                ) : null}
              </div>

              <div className="flex items-center gap-2">
                {editing ? (
                  <>
                    <button
                      type="button"
                      onClick={() => setEditing(false)}
                      className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
                    >
                      {t('common.cancel')}
                    </button>
                    <button
                      type="button"
                      disabled={!content.trim() || patchMutation.isPending}
                      onClick={() => patchMutation.mutate()}
                      className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent disabled:opacity-50"
                    >
                      {patchMutation.isPending ? t('common.loading') : t('common.save')}
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => {
                        setEditing(true)
                        setContent(memory.content ?? '')
                        setRole(memory.role ?? 'global')
                        setType(memory.type ?? 'manual_note')
                      }}
                      className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
                    >
                      {t('common.edit')}
                    </button>
                    {memory.status === 'active' ? (
                      <button
                        type="button"
                        disabled={archiveMutation.isPending}
                        onClick={() => archiveMutation.mutate()}
                        className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent disabled:opacity-50"
                      >
                        {t('common.archive')}
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled={activateMutation.isPending}
                        onClick={() => activateMutation.mutate()}
                        className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent disabled:opacity-50"
                      >
                        {t('common.activate')}
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>

            <div className="mt-2 text-[11px] text-muted">
              {t('run.createdAt')}: {formatTs(memory.created_at)} · {t('common.updatedAt')}: {formatTs(memory.updated_at)}
            </div>

            {editing ? (
              <div className="mt-3 grid gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-xs text-muted">{t('memories.role')}</label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg outline-none focus:border-accent"
                    disabled={!canEdit}
                  >
                    <option value="global">global</option>
                    <option value="orchestrator">orchestrator</option>
                    <option value="mof_expert">mof_expert</option>
                    <option value="tio2_expert">tio2_expert</option>
                  </select>
                  <label className="text-xs text-muted">{t('memories.type')}</label>
                  <select
                    value={type}
                    onChange={(e) => setType(e.target.value)}
                    className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg outline-none focus:border-accent"
                    disabled={!canEdit}
                  >
                    <option value="reasoningbank_item">reasoningbank_item</option>
                    <option value="manual_note">manual_note</option>
                  </select>
                </div>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  className="min-h-40 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-fg outline-none focus:border-accent"
                />
                {patchMutation.error ? (
                  <div className="text-sm text-danger">
                    {t('common.error')}: {(patchMutation.error as Error).message}
                  </div>
                ) : null}
              </div>
            ) : (
              <MemoryContentViewer memory={memory} />
            )}
          </section>

          <section className="rounded-lg border border-border bg-surface p-4">
            <div className="mb-2 text-sm font-medium text-fg">extra</div>
            <JsonViewer value={memory.extra ?? {}} defaultMode="tree" />
          </section>
        </>
      )}
    </div>
  )
}
