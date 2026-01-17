import { useMemo, useState } from 'react'
import { useT } from '../i18n/i18n'
import { Markdown } from './Markdown'

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

export function RichTextViewer(props: {
  text: string
  placeholder?: string
  maxHeightClass?: string
  defaultMode?: 'rendered' | 'raw'
}) {
  const t = useT()
  const [mode, setMode] = useState<'rendered' | 'raw'>(props.defaultMode ?? 'rendered')
  const [search, setSearch] = useState('')
  const [copied, setCopied] = useState(false)

  const text = props.text ?? ''
  const placeholder = props.placeholder ?? t('common.searchPlaceholder')
  const maxH = props.maxHeightClass ?? 'max-h-[520px]'

  const highlighted = useMemo(() => highlightText(text, search), [search, text])

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setMode('rendered')}
            className={`rounded-md border px-2 py-1 text-xs ${
              mode === 'rendered' ? 'border-accent text-accent' : 'border-border text-fg hover:border-accent'
            }`}
          >
            Rendered
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
          {mode === 'raw' ? (
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={placeholder}
              className="h-8 w-56 rounded-md border border-border bg-bg px-2 text-xs text-fg"
            />
          ) : null}
          <button
            type="button"
            onClick={async () => {
              const ok = await copyText(text)
              setCopied(ok)
              setTimeout(() => setCopied(false), 1200)
            }}
            className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-fg hover:border-accent"
          >
            {copied ? t('common.copied') : t('common.copy')}
          </button>
        </div>
      </div>

      {mode === 'raw' ? (
        <pre
          className={`${maxH} overflow-auto whitespace-pre-wrap rounded-md border border-border bg-surface p-3 text-xs text-fg`}
        >
          {highlighted}
        </pre>
      ) : (
        <div className={`${maxH} overflow-auto rounded-md border border-border bg-surface p-3`}>
          <Markdown text={text} />
        </div>
      )}
    </div>
  )
}

