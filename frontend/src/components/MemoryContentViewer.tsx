import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { MemoryItem } from '../api/types'
import { JsonViewer } from './JsonViewer'
import { Markdown } from './Markdown'
import { RichTextViewer } from './RichTextViewer'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readStringField(obj: unknown, key: string): string | null {
  if (!isRecord(obj)) return null
  const v = obj[key]
  return typeof v === 'string' ? v : null
}

function readNumberField(obj: unknown, key: string): number | null {
  if (!isRecord(obj)) return null
  const v = obj[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function readRecordField(obj: unknown, key: string): Record<string, unknown> | null {
  if (!isRecord(obj)) return null
  const v = obj[key]
  return isRecord(v) ? v : null
}

function readArrayField(obj: unknown, key: string): unknown[] | null {
  if (!isRecord(obj)) return null
  const v = obj[key]
  return Array.isArray(v) ? v : null
}

function snippet(text: string, n: number): string {
  const s = (text ?? '').replace(/\s+/g, ' ').trim()
  if (s.length <= n) return s
  return `${s.slice(0, n)}…`
}

type ParsedRbmemClaimsV1 = {
  format: 'RBMEM_CLAIMS_V1'
  header: Record<string, string>
  claims_json_raw: string
  claims: Record<string, unknown>[]
}

function parseRbmemClaimsV1(content: string): ParsedRbmemClaimsV1 | null {
  const s = (content ?? '').trim()
  if (!s.startsWith('RBMEM_CLAIMS_V1')) return null

  const lines = s.split(/\r?\n/)
  const idx = lines.findIndex((l) => l.startsWith('CLAIMS_JSON='))
  if (idx < 0) return null

  const header: Record<string, string> = {}
  for (const line of lines.slice(1, idx)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const eq = trimmed.indexOf('=')
    if (eq <= 0) continue
    const k = trimmed.slice(0, eq).trim()
    const v = trimmed.slice(eq + 1).trim()
    if (k) header[k] = v
  }

  let claimsRaw = lines[idx].slice('CLAIMS_JSON='.length)
  if (idx + 1 < lines.length) claimsRaw += `\n${lines.slice(idx + 1).join('\n')}`
  claimsRaw = claimsRaw.trim()

  try {
    const parsed = JSON.parse(claimsRaw)
    if (!Array.isArray(parsed)) return null
    const claims = parsed.filter((x) => isRecord(x)).map((x) => x as Record<string, unknown>)
    return { format: 'RBMEM_CLAIMS_V1', header, claims_json_raw: claimsRaw, claims }
  } catch {
    return null
  }
}

function statusBadgeClass(status: string): string {
  const s = (status ?? '').toLowerCase()
  if (s === 'fact') return 'border-success text-success'
  if (s === 'hypothesis') return 'border-warn text-warn'
  if (s === 'conclusion') return 'border-accent text-accent'
  return 'border-border text-muted'
}

function extractClaimId(claim: Record<string, unknown>): string {
  return (
    readStringField(claim, 'claim_id') ??
    readStringField(claim, 'id') ??
    readStringField(claim, 'claimId') ??
    '—'
  )
}

function extractClaimStatus(claim: Record<string, unknown>): string {
  return readStringField(claim, 'status') ?? 'unknown'
}

function extractClaimSummary(claim: Record<string, unknown>): string {
  const inf = readRecordField(claim, 'inference')
  const s = readStringField(inf, 'summary') ?? readStringField(claim, 'summary') ?? readStringField(claim, 'text')
  return (s ?? '').trim()
}

function extractSupportCount(claim: Record<string, unknown>): number {
  const support = readRecordField(claim, 'support')
  return readNumberField(support, 'count') ?? 0
}

function extractContraCount(claim: Record<string, unknown>): number {
  const contra = readRecordField(claim, 'contra')
  return readNumberField(contra, 'count') ?? 0
}

function extractFacts(claim: Record<string, unknown>): Record<string, unknown>[] {
  const facts = readArrayField(claim, 'facts') ?? []
  return facts.filter((x) => isRecord(x)).map((x) => x as Record<string, unknown>)
}

function FieldRow(props: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 rounded-md border border-border bg-bg px-3 py-2">
      <div className="font-mono text-[11px] text-muted">{props.k}</div>
      <div className="min-w-0 text-[11px] text-fg">{props.v}</div>
    </div>
  )
}

function ClaimDetails(props: { claim: Record<string, unknown> }) {
  const claimId = extractClaimId(props.claim)
  const status = extractClaimStatus(props.claim)
  const summary = extractClaimSummary(props.claim)
  const facts = extractFacts(props.claim)
  const support = extractSupportCount(props.claim)
  const contra = extractContraCount(props.claim)

  const constraint = readRecordField(readRecordField(props.claim, 'inference'), 'constraint')
  const constraintEntries = constraint ? Object.entries(constraint) : []

  const supportRunIds = readArrayField(readRecordField(props.claim, 'support'), 'run_ids') ?? []
  const contraRunIds = readArrayField(readRecordField(props.claim, 'contra'), 'run_ids') ?? []

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="font-mono text-sm text-fg">{claimId}</div>
            <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusBadgeClass(status)}`}>
              {status}
            </span>
          </div>
          <div className="mt-1 text-[11px] text-muted">
            facts={facts.length} · support={support} · contra={contra}
          </div>
        </div>
      </div>

      {summary ? (
        <div className="rounded-md border border-border bg-surface p-3">
          <div className="mb-2 text-[11px] font-medium text-muted">summary</div>
          <Markdown text={summary} />
        </div>
      ) : null}

      {constraintEntries.length ? (
        <div className="grid gap-2">
          <div className="text-[11px] font-medium text-muted">constraints</div>
          <div className="grid gap-2">
            {constraintEntries.map(([k, v]) => (
              <FieldRow key={k} k={k} v={<span className="font-mono">{String(v)}</span>} />
            ))}
          </div>
        </div>
      ) : null}

      {supportRunIds.length || contraRunIds.length ? (
        <div className="grid gap-2">
          <div className="text-[11px] font-medium text-muted">evidence</div>
          <div className="grid gap-2">
            {supportRunIds.length ? (
              <FieldRow
                k="support_runs"
                v={
                  <div className="flex flex-wrap justify-end gap-2">
                    {supportRunIds.slice(0, 10).map((rid) => (
                      <Link
                        key={String(rid)}
                        to={`/runs/${encodeURIComponent(String(rid))}`}
                        className="rounded-md border border-border bg-bg px-2 py-0.5 font-mono text-[11px] text-fg hover:border-accent"
                      >
                        run:{String(rid)}
                      </Link>
                    ))}
                    {supportRunIds.length > 10 ? (
                      <span className="text-[11px] text-muted">… +{supportRunIds.length - 10}</span>
                    ) : null}
                  </div>
                }
              />
            ) : null}

            {contraRunIds.length ? (
              <FieldRow
                k="contra_runs"
                v={
                  <div className="flex flex-wrap justify-end gap-2">
                    {contraRunIds.slice(0, 10).map((rid) => (
                      <Link
                        key={String(rid)}
                        to={`/runs/${encodeURIComponent(String(rid))}`}
                        className="rounded-md border border-border bg-bg px-2 py-0.5 font-mono text-[11px] text-fg hover:border-accent"
                      >
                        run:{String(rid)}
                      </Link>
                    ))}
                    {contraRunIds.length > 10 ? (
                      <span className="text-[11px] text-muted">… +{contraRunIds.length - 10}</span>
                    ) : null}
                  </div>
                }
              />
            ) : null}
          </div>
        </div>
      ) : null}

      {facts.length ? (
        <div className="grid gap-2">
          <div className="text-[11px] font-medium text-muted">facts</div>
          <div className="grid gap-2">
            {facts.map((f, idx) => {
              const runId = readStringField(f, 'run_id')
              const recipes = readArrayField(f, 'recipes')
              const recipeCount = recipes ? recipes.length : null
              return (
                <div key={idx} className="rounded-md border border-border bg-bg p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-xs font-medium text-fg">fact #{idx + 1}</div>
                    {runId ? (
                      <Link
                        to={`/runs/${encodeURIComponent(runId)}`}
                        className="rounded-md border border-border bg-surface px-2 py-0.5 font-mono text-[11px] text-fg hover:border-accent"
                      >
                        run:{runId}
                      </Link>
                    ) : null}
                  </div>
                  <div className="mt-1 text-[11px] text-muted">
                    {recipeCount !== null ? `recipes=${recipeCount}` : null}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}

      <div className="rounded-md border border-border bg-bg p-3">
        <div className="mb-2 text-[11px] font-medium text-muted">raw_claim</div>
        <JsonViewer value={props.claim} defaultMode="tree" />
      </div>
    </div>
  )
}

export function MemoryContentViewer(props: {
  memory: Pick<MemoryItem, 'content' | 'type' | 'extra'>
}) {
  const content = props.memory.content ?? ''
  const parsed = useMemo(() => parseRbmemClaimsV1(content), [content])

  const defaultMode: 'structured' | 'raw' = parsed ? 'structured' : 'raw'
  const [mode, setMode] = useState<'structured' | 'raw'>(defaultMode)
  const [q, setQ] = useState('')

  const claims = parsed?.claims ?? []
  const claimsById = useMemo(() => {
    const m = new Map<string, Record<string, unknown>>()
    for (const c of claims) m.set(extractClaimId(c), c)
    return m
  }, [claims])

  const [selectedId, setSelectedId] = useState<string | null>(() => (claims[0] ? extractClaimId(claims[0]) : null))

  const filtered = useMemo(() => {
    if (!parsed) return []
    const query = q.trim().toLowerCase()
    if (!query) return claims

    const out: Record<string, unknown>[] = []
    for (const c of claims) {
      const id = extractClaimId(c).toLowerCase()
      const st = extractClaimStatus(c).toLowerCase()
      const summary = extractClaimSummary(c).toLowerCase()
      if (id.includes(query) || st.includes(query) || summary.includes(query)) out.push(c)
    }
    return out
  }, [claims, parsed, q])

  const selectedClaim = selectedId ? claimsById.get(selectedId) ?? null : null

  const notesFromExtra = useMemo(() => {
    const notes = isRecord(props.memory.extra) ? props.memory.extra['notes'] : null
    return typeof notes === 'string' ? notes.trim() : ''
  }, [props.memory.extra])

  if (!parsed) {
    // Default: treat memory content as rich text for readability (markdown, safe HTML, math).
    return (
      <div className="mt-3">
        <RichTextViewer text={content} defaultMode="rendered" />
      </div>
    )
  }

  const statusCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const c of claims) {
      const st = extractClaimStatus(c) || 'unknown'
      m.set(st, (m.get(st) ?? 0) + 1)
    }
    return m
  }, [claims])

  return (
    <div className="mt-3 grid gap-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-semibold text-fg">RB Claims (v1)</div>
            <span className="rounded-full border border-border bg-bg px-2 py-0.5 text-[11px] text-muted">
              claims={claims.length}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-muted">
            {[...statusCounts.entries()].map(([st, n]) => (
              <span key={st} className={`rounded-full border px-2 py-0.5 ${statusBadgeClass(st)}`}>
                {st}:{n}
              </span>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setMode('structured')}
            className={`rounded-md border px-2 py-1 text-xs ${
              mode === 'structured' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            Structured
          </button>
          <button
            type="button"
            onClick={() => setMode('raw')}
            className={`rounded-md border px-2 py-1 text-xs ${
              mode === 'raw' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            Raw
          </button>
        </div>
      </div>

      {notesFromExtra ? (
        <div className="rounded-md border border-border bg-bg p-3">
          <div className="mb-1 text-[11px] font-medium text-muted">notes</div>
          <div className="text-xs text-fg">{notesFromExtra}</div>
        </div>
      ) : null}

      {mode === 'raw' ? (
        <RichTextViewer text={content} defaultMode="raw" />
      ) : (
        <div className="grid gap-4 md:grid-cols-[320px_1fr]">
          <div className="rounded-md border border-border bg-bg p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs font-medium text-muted">claims</div>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search…"
                className="h-8 w-40 rounded-md border border-border bg-bg px-2 text-xs text-fg"
              />
            </div>
            {filtered.length === 0 ? (
              <div className="text-sm text-muted">No matches</div>
            ) : (
              <div className="grid gap-1">
                {filtered.map((c) => {
                  const id = extractClaimId(c)
                  const st = extractClaimStatus(c)
                  const summary = extractClaimSummary(c)
                  const facts = extractFacts(c)
                  const support = extractSupportCount(c)
                  const contra = extractContraCount(c)
                  const selected = selectedId === id
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => setSelectedId(id)}
                      className={`w-full rounded-md border px-2 py-2 text-left ${
                        selected ? 'border-accent bg-surface' : 'border-border bg-bg hover:border-accent'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-mono text-xs text-fg">{id}</div>
                        <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusBadgeClass(st)}`}>
                          {st}
                        </span>
                      </div>
                      {summary ? (
                        <div className="mt-1 text-[11px] text-muted">{snippet(summary, 120)}</div>
                      ) : (
                        <div className="mt-1 text-[11px] text-muted">—</div>
                      )}
                      <div className="mt-1 text-[10px] text-muted">
                        facts={facts.length} · support={support} · contra={contra}
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <div className="rounded-md border border-border bg-bg p-3">
            {selectedClaim ? <ClaimDetails claim={selectedClaim} /> : <div className="text-sm text-muted">—</div>}
          </div>
        </div>
      )}
    </div>
  )
}

