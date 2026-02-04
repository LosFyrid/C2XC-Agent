import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'

// NOTE: Evidence/trace text is treated as untrusted input (KB chunks, LLM outputs).
// We allow a small subset of HTML (e.g. <sub>/<sup>) but sanitize to avoid XSS.
const SANITIZE_SCHEMA = (() => {
  const base = defaultSchema as unknown as {
    tagNames?: string[]
    attributes?: Record<string, unknown>
  }

  const tagNames = new Set([...(base.tagNames ?? [])])
  tagNames.add('sub')
  tagNames.add('sup')
  tagNames.add('br')

  // Avoid loading remote resources via markdown/HTML.
  tagNames.delete('img')

  // Keep class names for math blocks created by `remark-math` (rehype-katex looks for
  // `language-math`, `math-inline`, `math-display`).
  const attributes = { ...(base.attributes ?? {}) } as Record<string, unknown>
  attributes.span = [...new Set([...(attributes.span as string[] | undefined) ?? [], 'className'])]
  attributes.div = [...new Set([...(attributes.div as string[] | undefined) ?? [], 'className'])]
  attributes.pre = [...new Set([...(attributes.pre as string[] | undefined) ?? [], 'className'])]
  attributes.code = [
    ...new Set([
      ...((attributes.code as unknown[] | undefined) ?? []),
      // Allow className (including `math-inline` etc); we style code blocks ourselves anyway.
      'className',
    ]),
  ]

  return {
    ...base,
    tagNames: [...tagNames],
    attributes,
  }
})()

function preprocessMarkdown(text: string): string {
  const s = text ?? ''

  // Some KB chunks contain double-escaped LaTeX commands inside `$...$` (e.g. `$\\mu$`).
  // Normalize those to valid KaTeX input (`$\mu$`) while leaving non-math escapes intact.
  let out = ''
  let i = 0
  while (i < s.length) {
    const ch = s[i]

    // Skip escaped characters (notably `\$` for literal `$`).
    if (ch === '\\' && i + 1 < s.length) {
      out += s.slice(i, i + 2)
      i += 2
      continue
    }

    if (ch === '$') {
      const isDisplay = s[i + 1] === '$'
      const delimLen = isDisplay ? 2 : 1
      const delim = isDisplay ? '$$' : '$'
      const start = i + delimLen

      let j = start
      while (j < s.length) {
        const cj = s[j]
        if (cj === '\\' && j + 1 < s.length) {
          j += 2
          continue
        }
        if (cj === '$') {
          if (isDisplay) {
            if (s[j + 1] === '$') break
          } else {
            break
          }
        }
        j += 1
      }

      // Unmatched `$` / `$$`: treat as plain text.
      if (j >= s.length) {
        out += ch
        i += 1
        continue
      }

      const inner = s.slice(start, j)
      const normalizedInner = inner.replace(/\\\\/g, '\\')
      out += delim + normalizedInner + delim
      i = j + delimLen
      continue
    }

    out += ch
    i += 1
  }

  return out
}

export function Markdown(props: { text: string }) {
  const text = preprocessMarkdown(props.text ?? '')

  return (
    <div className="grid gap-2 text-sm text-fg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, SANITIZE_SCHEMA], rehypeKatex]}
        components={{
          p: ({ children }) => <p className="whitespace-pre-wrap leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline underline-offset-2 hover:opacity-90"
            >
              {children}
            </a>
          ),
          code: ({ children }) => (
            <code className="rounded bg-surface px-1 py-0.5 font-mono text-[0.95em]">{children}</code>
          ),
          pre: ({ children }) => (
            <pre className="overflow-auto rounded-md border border-border bg-surface p-3 text-xs">{children}</pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-border pl-3 text-muted">{children}</blockquote>
          ),
          hr: () => <hr className="border-border" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
