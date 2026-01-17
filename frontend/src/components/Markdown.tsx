import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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

  return {
    ...base,
    tagNames: [...tagNames],
  }
})()

function preprocessMarkdown(text: string): string {
  const s = text ?? ''

  // Minimal math-ish conveniences seen in the KB/LLM outputs. We do not try to
  // implement full LaTeX rendering here (keeps dependencies + attack surface small).
  return (
    s
      .replaceAll('$\\rightarrow$', '→')
      // Only replace standalone commands to avoid mangling e.g. "\tool" -> "→ol".
      .replace(/\\rightarrow(?![A-Za-z])/g, '→')
      .replaceAll('$\\to$', '→')
      .replace(/\\to(?![A-Za-z])/g, '→')
  )
}

export function Markdown(props: { text: string }) {
  const text = preprocessMarkdown(props.text ?? '')

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
