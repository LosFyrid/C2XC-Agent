import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import { Link } from 'react-router-dom'

// Matches:
// - citation aliases like [C12]
// - memory ids like mem:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
const TOKEN_RE =
  /\[(?<alias>[A-Z]+\d+)\]|mem:(?<memId>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})/g

const SANITIZE_SCHEMA = (() => {
  const base = defaultSchema as unknown as {
    tagNames?: string[]
    protocols?: Record<string, unknown>
  }
  const tagNames = new Set([...(base.tagNames ?? [])])
  tagNames.add('sub')
  tagNames.add('sup')
  tagNames.add('br')
  tagNames.delete('img')
  const protocols = { ...(base.protocols ?? {}) } as Record<string, unknown>
  const href = new Set([...(protocols.href as string[] | undefined) ?? []])
  href.add('c2xc')
  protocols.href = [...href]

  return { ...base, tagNames: [...tagNames], protocols }
})()

function preprocessMarkdown(text: string): string {
  const s = text ?? ''
  return (
    s
      .replaceAll('$\\rightarrow$', '→')
      .replace(/\\rightarrow(?![A-Za-z])/g, '→')
      .replaceAll('$\\to$', '→')
      .replace(/\\to(?![A-Za-z])/g, '→')
  )
}

function injectC2xcLinks(
  text: string,
  knownAliases?: Set<string>,
  knownMemIds?: Set<string>,
): string {
  const s = text ?? ''

  let out = ''
  let lastIndex = 0
  for (const match of s.matchAll(TOKEN_RE)) {
    const index = match.index ?? 0
    const rawAlias = (match.groups?.alias ?? '').trim()
    const rawMemId = (match.groups?.memId ?? '').trim()
    if (index > lastIndex) out += s.slice(lastIndex, index)

    const token = match[0] ?? ''
    const isKnownAlias = rawAlias && (!knownAliases || knownAliases.has(rawAlias))
    const isKnownMem = rawMemId && (!knownMemIds || knownMemIds.has(rawMemId))

    if (rawAlias && isKnownAlias) {
      // Keep the visible label as "[C12]" but represent it as a markdown link.
      out += `[${token}](c2xc://citation/${encodeURIComponent(rawAlias)})`
    } else if (rawMemId && isKnownMem) {
      out += `[mem:${rawMemId}](c2xc://mem/${encodeURIComponent(rawMemId)})`
    } else {
      out += token
    }

    lastIndex = index + token.length
  }
  if (lastIndex < s.length) out += s.slice(lastIndex)
  return out
}

export function CitationMarkdown(props: {
  text: string
  knownAliases?: Set<string>
  onClickAlias?: (alias: string) => void
  knownMemIds?: Set<string>
  onClickMemId?: (memId: string) => void
}) {
  const text = preprocessMarkdown(injectC2xcLinks(props.text ?? '', props.knownAliases, props.knownMemIds))
  const onClickAlias = props.onClickAlias
  const onClickMemId = props.onClickMemId

  return (
    <div className="grid gap-2 text-sm text-fg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, SANITIZE_SCHEMA]]}
        components={{
          p: ({ children }) => <p className="whitespace-pre-wrap leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          a: ({ children, href }) => {
            const h = String(href ?? '')
            if (h.startsWith('c2xc://citation/')) {
              const alias = decodeURIComponent(h.slice('c2xc://citation/'.length))
              if (!onClickAlias) {
                return <span className="font-mono">{children as ReactNode}</span>
              }
              return (
                <button
                  type="button"
                  onClick={() => onClickAlias(alias)}
                  className="mx-0.5 inline-flex items-baseline rounded-sm px-0.5 font-mono text-accent underline underline-offset-2 hover:opacity-90"
                  title={`Open evidence [${alias}]`}
                >
                  {children as ReactNode}
                </button>
              )
            }
            if (h.startsWith('c2xc://mem/')) {
              const memId = decodeURIComponent(h.slice('c2xc://mem/'.length))
              if (onClickMemId) {
                return (
                  <button
                    type="button"
                    onClick={() => onClickMemId(memId)}
                    className="mx-0.5 inline-flex items-baseline rounded-sm px-0.5 font-mono text-accent underline underline-offset-2 hover:opacity-90"
                    title={`Open memory mem:${memId}`}
                  >
                    {children as ReactNode}
                  </button>
                )
              }
              return (
                <Link
                  to={`/memories/${encodeURIComponent(memId)}`}
                  className="mx-0.5 inline-flex items-baseline rounded-sm px-0.5 font-mono text-accent underline underline-offset-2 hover:opacity-90"
                  title={`Open memory mem:${memId}`}
                >
                  {children as ReactNode}
                </Link>
              )
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-accent underline underline-offset-2 hover:opacity-90"
              >
                {children as ReactNode}
              </a>
            )
          },
          code: ({ children }) => (
            <code className="rounded bg-surface px-1 py-0.5 font-mono text-[0.95em]">{children}</code>
          ),
          pre: ({ children }) => (
            <pre className="overflow-auto rounded-md border border-border bg-surface p-3 text-xs">{children}</pre>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
