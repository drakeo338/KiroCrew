/**
 * Quote a WHOLE message (not a selection) into the next send.
 *
 * The quote travels two ways at once, and both come from ONE record:
 *
 * - As TEXT, for the agent: the quoted message is prepended to the typed text
 *   as a markdown blockquote with an attribution line (`quoteBlock`). That is
 *   the only form every consumer of the transcript sees — the model, a channel
 *   relay, a raw export — so the quote is never something only the dashboard
 *   knows about.
 * - As META, for the dashboard: `meta.quote` carries the same record, and the
 *   user bubble draws it as a card (author · time · excerpt, click = jump to the
 *   quoted message) INSTEAD of the raw `>` lines, which `stripQuoteBlock`
 *   removes from the rendered body. Same pattern as collapsed pastes
 *   (`[ Paste #N ]` tokens + `meta.pastes`) and staged files.
 *
 * Because the block is a pure function of the record, the strip is exact: a
 * row whose content does not begin with `quoteBlock(meta.quote)` (edited by
 * hand, a foreign client) renders its content untouched and keeps the card —
 * the card is additive, never a reason to hide text.
 *
 * One quote per message. Quoting again replaces the staged quote rather than
 * stacking; stacking is what the selection quote (`quoteIntoDraft`) is for.
 */

export type MessageQuoteRole = 'user' | 'assistant'

export interface MessageQuote {
  /** Who wrote the quoted message. Drives the card's author label. */
  role: MessageQuoteRole
  /** The quoted text, trimmed and capped (`QUOTE_TEXT_MAX`). */
  text: string
  /** Server ts of the quoted message; the jump target. Absent for a row with none. */
  ts?: string
  /** Stable message id, preferred over `ts` when jumping (a same-ts pair). */
  mid?: string
}

/** Cap on the quoted text. Long enough to carry a whole ordinary reply to the
 *  agent verbatim; short enough that quoting a 20 KB dump does not double the
 *  turn. Past it the text is cut at a word and marked with an ellipsis. */
export const QUOTE_TEXT_MAX = 1500

/** Paragraph break between the block and the typed text: leaves the caret on
 *  its own line, and lets `stripQuoteBlock` find the boundary exactly. */
const BLANK_LINE = '\n\n'

function capText(text: string): string {
  const trimmed = text.trim()
  if (trimmed.length <= QUOTE_TEXT_MAX) return trimmed
  const cut = trimmed.slice(0, QUOTE_TEXT_MAX)
  const atWord = cut.lastIndexOf(' ')
  return (atWord > QUOTE_TEXT_MAX * 0.6 ? cut.slice(0, atWord) : cut).trimEnd() + '…'
}

/** Build the record for a transcript row. `null` for a row with nothing to quote. */
export function quoteFromMessage(role: MessageQuoteRole, content: string, ts?: string, mid?: string): MessageQuote | null {
  const text = capText(content)
  if (!text) return null
  const q: MessageQuote = { role, text }
  if (ts) q.ts = ts
  if (mid) q.mid = mid
  return q
}

/** The attribution line inside the block. English on purpose: this is the form
 *  the agent reads, in the transcript's own language of record; the card the
 *  user sees carries the localized author label instead. */
function attribution(role: MessageQuoteRole): string {
  return role === 'user' ? '— quoting an earlier message from the user' : '— quoting an earlier message from the assistant'
}

/** The markdown blockquote the record serializes to. */
export function quoteBlock(q: MessageQuote): string {
  return [...q.text.split('\n'), attribution(q.role)].map(line => '> ' + line).join('\n')
}

/** The full message text for a send: block, blank line, what was typed. */
export function prependQuote(typed: string, q: MessageQuote): string {
  const body = typed.trim()
  return body ? quoteBlock(q) + BLANK_LINE + body : quoteBlock(q)
}

/** The body to RENDER for a row carrying `meta.quote`: the block removed when
 *  (and only when) the content begins with exactly that block. */
export function stripQuoteBlock(content: string, q: MessageQuote): string {
  const block = quoteBlock(q)
  if (!content.startsWith(block)) return content
  return content.slice(block.length).replace(/^\n+/, '')
}

/** Read `meta.quote` off a row, refusing any shape a card could not draw. */
export function readMessageQuote(meta: Record<string, unknown> | undefined): MessageQuote | null {
  const raw = meta?.quote
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  if ((r.role !== 'user' && r.role !== 'assistant') || typeof r.text !== 'string' || !r.text.trim()) return null
  const q: MessageQuote = { role: r.role, text: r.text }
  if (typeof r.ts === 'string' && r.ts) q.ts = r.ts
  if (typeof r.mid === 'string' && r.mid) q.mid = r.mid
  return q
}
