import { useMemo, useState } from 'react'
import { RichTextViewer } from './RichTextViewer'

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch (e) {
    return `<<json stringify error: ${(e as Error).message}>>`
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function highlightText(text: string, query: string): React.ReactNode {
  const q = query.trim()
  if (!q) return text
  if (q.length > 64) return text

  const re = new RegExp(escapeRegExp(q), 'gi')
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  for (const m of text.matchAll(re)) {
    const index = m.index ?? 0
    const token = m[0] ?? ''
    if (index > lastIndex) parts.push(text.slice(lastIndex, index))
    parts.push(
      <mark key={`${index}_${token}`} className="rounded-sm bg-warn px-0.5 text-fg">
        {token}
      </mark>,
    )
    lastIndex = index + token.length
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return <>{parts}</>
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

export function JsonViewer(props: { value: unknown; defaultMode?: 'tree' | 'raw' }) {
  const [mode, setMode] = useState<'tree' | 'raw'>(props.defaultMode ?? 'tree')
  const [search, setSearch] = useState('')
  const [copied, setCopied] = useState(false)

  const jsonText = useMemo(() => safeStringify(props.value), [props.value])

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setMode('tree')}
            className={`rounded-md border px-2 py-1 text-xs ${
              mode === 'tree' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            Tree
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

        <div className="flex items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search…"
            className="h-8 w-48 rounded-md border border-border bg-bg px-2 text-xs text-fg"
          />
          <button
            type="button"
            onClick={async () => {
              const ok = await copyText(jsonText)
              setCopied(ok)
              setTimeout(() => setCopied(false), 1200)
            }}
            className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {mode === 'raw' ? (
        <pre className="max-h-[520px] overflow-auto whitespace-pre-wrap rounded-md border border-border bg-bg p-3 text-xs">
          {highlightText(jsonText, search)}
        </pre>
      ) : (
        <TreeView value={props.value} search={search} />
      )}
    </div>
  )
}

function TreeView(props: { value: unknown; search: string }) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ root: true })

  const toggle = (path: string) => {
    setExpanded((m) => ({ ...m, [path]: !m[path] }))
  }

  return (
    <div className="max-h-[520px] overflow-auto rounded-md border border-border bg-bg p-3 text-xs">
      <TreeNode
        value={props.value}
        name="root"
        path="root"
        depth={0}
        expanded={expanded}
        toggle={toggle}
        search={props.search}
      />
    </div>
  )
}

function TreeNode(props: {
  value: unknown
  name: string
  path: string
  depth: number
  expanded: Record<string, boolean>
  toggle: (path: string) => void
  search: string
}) {
  const { value, name, path, depth, expanded, toggle, search } = props
  const indent = { paddingLeft: `${depth * 14}px` }

  const isArray = Array.isArray(value)
  const isObj = isRecord(value)
  const isRichString =
    typeof value === 'string' &&
    (value.length > 120 ||
      value.includes('\n') ||
      value.includes('<sub') ||
      value.includes('<sup') ||
      value.includes('$\\') ||
      value.includes('\\rightarrow') ||
      value.includes('\\to'))
  const isExpandable = isArray || isObj || isRichString
  const open = !!expanded[path]

  const label = useMemo(() => {
    if (isArray) return `${name}: Array(${value.length})`
    if (isObj) return `${name}: Object(${Object.keys(value).length})`
    if (typeof value === 'string') {
      const maxPreview = 120
      const preview = value.length > maxPreview ? value.slice(0, maxPreview) + '…' : value
      const meta = value.length > maxPreview ? ` (${value.length} chars)` : ''
      return `${name}: "${preview}"${meta}`
    }
    if (value === null) return `${name}: null`
    return `${name}: ${String(value)}`
  }, [isArray, isObj, name, value])

  const keyMatches = search.trim() ? name.toLowerCase().includes(search.trim().toLowerCase()) : false

  return (
    <div>
      <div className="flex items-start gap-2" style={indent}>
        {isExpandable ? (
          <button
            type="button"
            onClick={() => toggle(path)}
            className="mt-0.5 inline-flex h-4 w-4 items-center justify-center rounded border border-border bg-surface font-mono text-[10px] text-fg hover:border-accent"
            title={open ? 'Collapse' : 'Expand'}
          >
            {open ? '−' : '+'}
          </button>
        ) : (
          <span className="mt-0.5 inline-flex h-4 w-4 items-center justify-center text-[10px] text-muted">
            ·
          </span>
        )}
        <div className={`break-words ${keyMatches ? 'text-accent' : 'text-fg'}`}>{label}</div>
      </div>

      {isExpandable && open ? (
        isArray ? (
          <TreeArray value={value} path={path} depth={depth} expanded={expanded} toggle={toggle} search={search} />
        ) : isObj ? (
          <TreeObject value={value} path={path} depth={depth} expanded={expanded} toggle={toggle} search={search} />
        ) : (
          <TreeString value={String(value ?? '')} depth={depth} />
        )
      ) : null}
    </div>
  )
}

function TreeString(props: { value: string; depth: number }) {
  const indent = { paddingLeft: `${(props.depth + 1) * 14}px` }
  return (
    <div className="mt-1" style={indent}>
      <RichTextViewer text={props.value} maxHeightClass="max-h-[420px]" />
    </div>
  )
}

function TreeObject(props: {
  value: Record<string, unknown>
  path: string
  depth: number
  expanded: Record<string, boolean>
  toggle: (path: string) => void
  search: string
}) {
  const defaultLimit = 80
  const [limit, setLimit] = useState(defaultLimit)

  const keys = Object.keys(props.value)
  const shown = keys.slice(0, limit)
  const remaining = keys.length - shown.length

  return (
    <div>
      {shown.map((k) => (
        <TreeNode
          key={`${props.path}.${k}`}
          value={props.value[k]}
          name={k}
          path={`${props.path}.${k}`}
          depth={props.depth + 1}
          expanded={props.expanded}
          toggle={props.toggle}
          search={props.search}
        />
      ))}
      {remaining > 0 ? (
        <div className="pl-6">
          <div className="text-muted">… {remaining} more keys hidden</div>
          <div className="mt-1 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setLimit((n) => Math.min(keys.length, n + defaultLimit))}
              className="rounded-md border border-border bg-bg px-2 py-1 text-[11px] text-fg hover:border-accent"
            >
              Show more
            </button>
            <button
              type="button"
              onClick={() => setLimit(keys.length)}
              className="rounded-md border border-border bg-bg px-2 py-1 text-[11px] text-fg hover:border-accent"
            >
              Show all
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function TreeArray(props: {
  value: unknown[]
  path: string
  depth: number
  expanded: Record<string, boolean>
  toggle: (path: string) => void
  search: string
}) {
  const defaultLimit = 50
  const [limit, setLimit] = useState(defaultLimit)

  const shown = props.value.slice(0, limit)
  const remaining = props.value.length - shown.length

  return (
    <div>
      {shown.map((v, idx) => (
        <TreeNode
          key={`${props.path}[${idx}]`}
          value={v}
          name={`[${idx}]`}
          path={`${props.path}[${idx}]`}
          depth={props.depth + 1}
          expanded={props.expanded}
          toggle={props.toggle}
          search={props.search}
        />
      ))}
      {remaining > 0 ? (
        <div className="pl-6">
          <div className="text-muted">… {remaining} more items hidden</div>
          <div className="mt-1 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setLimit((n) => Math.min(props.value.length, n + defaultLimit))}
              className="rounded-md border border-border bg-bg px-2 py-1 text-[11px] text-fg hover:border-accent"
            >
              Show more
            </button>
            <button
              type="button"
              onClick={() => setLimit(props.value.length)}
              className="rounded-md border border-border bg-bg px-2 py-1 text-[11px] text-fg hover:border-accent"
            >
              Show all
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
