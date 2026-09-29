import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useMessageQuote } from '../chat-core/composer/useMessageQuote'
import { quoteBlock } from '../chat-core/composer/messageQuote'

describe('useMessageQuote', () => {
  it('stages a message, reveals the composer, and replaces rather than stacks', () => {
    const reveal = vi.fn()
    const { result } = renderHook(() => useMessageQuote({ slot: 'chat-1', revealComposer: reveal }))
    act(() => result.current.quoteMessage('assistant', 'first', 't1', 'm1'))
    expect(result.current.pendingQuote).toEqual({ role: 'assistant', text: 'first', ts: 't1', mid: 'm1' })
    expect(reveal).toHaveBeenCalledTimes(1)
    act(() => result.current.quoteMessage('user', 'second', 't2'))
    expect(result.current.pendingQuote).toEqual({ role: 'user', text: 'second', ts: 't2' })
  })

  it('ignores a row with nothing to quote', () => {
    const { result } = renderHook(() => useMessageQuote({ slot: 'chat-1' }))
    act(() => result.current.quoteMessage('user', '   '))
    expect(result.current.pendingQuote).toBeNull()
  })

  it('consume hands back the record with the block prepended and clears the stage', () => {
    const { result } = renderHook(() => useMessageQuote({ slot: 'chat-1' }))
    act(() => result.current.quoteMessage('assistant', 'quoted', 't1'))
    let out: ReturnType<typeof result.current.consume> | undefined
    act(() => { out = result.current.consume('typed') })
    expect(out!.quote).toEqual({ role: 'assistant', text: 'quoted', ts: 't1' })
    expect(out!.text).toBe(quoteBlock(out!.quote!) + '\n\ntyped')
    expect(result.current.pendingQuote).toBeNull()
  })

  it('consume with nothing staged returns the text untouched', () => {
    const { result } = renderHook(() => useMessageQuote({ slot: 'chat-1' }))
    let out: ReturnType<typeof result.current.consume> | undefined
    act(() => { out = result.current.consume('typed') })
    expect(out).toEqual({ quote: null, text: 'typed' })
  })

  it('restage puts a consumed quote back (a failed send); null is a no-op', () => {
    const { result } = renderHook(() => useMessageQuote({ slot: 'chat-1' }))
    act(() => result.current.quoteMessage('user', 'q', 't1'))
    let taken: ReturnType<typeof result.current.consume> | undefined
    act(() => { taken = result.current.consume('') })
    expect(result.current.pendingQuote).toBeNull()
    act(() => result.current.restage(taken!.quote))
    expect(result.current.pendingQuote).toEqual({ role: 'user', text: 'q', ts: 't1' })
    act(() => result.current.clearQuote())
    act(() => result.current.restage(null))
    expect(result.current.pendingQuote).toBeNull()
  })

  it('drops the stage on a slot switch, not on mount', () => {
    const { result, rerender } = renderHook(({ slot }: { slot: string | null }) => useMessageQuote({ slot }), { initialProps: { slot: 'chat-1' } })
    act(() => result.current.quoteMessage('user', 'q'))
    rerender({ slot: 'chat-1' })
    expect(result.current.pendingQuote).not.toBeNull()
    rerender({ slot: 'chat-2' })
    expect(result.current.pendingQuote).toBeNull()
  })
})
