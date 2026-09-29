import { useCallback, useEffect, useRef, useState } from 'react'
import { type MessageQuote, type MessageQuoteRole, prependQuote, quoteFromMessage } from './messageQuote'

/**
 * Host-side state for quoting a whole message into the next send.
 *
 * One quote at a time, per surface: quoting another message REPLACES the staged
 * one (the card in the composer shows which). The staged quote is scoped to the
 * slot it was taken from -- a slot switch drops it, since a quote of a message
 * in conversation A is meaningless as the opening of a send into B, and the
 * transcript it points at is no longer on screen to check.
 *
 * `consume()` is what `send()` calls: it hands back the record and the text to
 * send, and clears the stage in the same step so a failed send that restores
 * the composer does not silently re-quote (the host decides whether to
 * `restage` it, exactly as it decides for files and session refs).
 */
export interface UseMessageQuote {
  pendingQuote: MessageQuote | null
  /** Stage `content` as the quote. A row with nothing quotable is a no-op. */
  quoteMessage: (role: MessageQuoteRole, content: string, ts?: string, mid?: string) => void
  clearQuote: () => void
  /** Put a previously consumed quote back (a failed send). */
  restage: (quote: MessageQuote | null) => void
  /** Take the staged quote for a send: `{ quote, text }` with the block
   *  prepended to `typed`, or `{ quote: null, text: typed }` when none. */
  consume: (typed: string) => { quote: MessageQuote | null; text: string }
}

export function useMessageQuote({ slot, revealComposer }: {
  /** The slot the surface shows; the stage is dropped when it changes. */
  slot?: string | null
  /** Bring the composer into view once the quote is staged. */
  revealComposer?: () => void
}): UseMessageQuote {
  const [pendingQuote, setPendingQuote] = useState<MessageQuote | null>(null)
  const pendingRef = useRef<MessageQuote | null>(null)
  pendingRef.current = pendingQuote

  // Slot-scoped: drop the stage on a switch, not on mount.
  const lastSlot = useRef(slot)
  useEffect(() => {
    if (lastSlot.current !== slot) {
      lastSlot.current = slot
      setPendingQuote(null)
    }
  }, [slot])

  const quoteMessage = useCallback((role: MessageQuoteRole, content: string, ts?: string, mid?: string) => {
    const q = quoteFromMessage(role, content, ts, mid)
    if (!q) return
    setPendingQuote(q)
    revealComposer?.()
  }, [revealComposer])
  const clearQuote = useCallback(() => setPendingQuote(null), [])
  const restage = useCallback((q: MessageQuote | null) => { if (q) setPendingQuote(q) }, [])
  const consume = useCallback((typed: string) => {
    const q = pendingRef.current
    if (!q) return { quote: null, text: typed }
    setPendingQuote(null)
    return { quote: q, text: prependQuote(typed, q) }
  }, [])

  return { pendingQuote, quoteMessage, clearQuote, restage, consume }
}
