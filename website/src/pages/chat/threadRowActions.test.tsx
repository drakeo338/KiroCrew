/**
 * Where the thread action lives on each message row, and what the row costs.
 *
 * The action row on both message kinds already carries more peer buttons than the
 * two-button rule allows. The rule lets a row in that position keep the count it
 * has but never grow, so a thread action cannot be one more button there. It is a
 * menu item on a finished row. The streaming reply is the exception and keeps a
 * visible control, because that row holds exactly one.
 */
import { describe, it, expect, vi } from 'vitest'

vi.mock('@radix-ui/react-dropdown-menu', async () => await import('../../test/__mocks__/@radix-ui/react-dropdown-menu'))

import { screen, fireEvent } from '@testing-library/react'
import { renderWithProviders } from '../../test/helpers'
import UserMessage from './UserMessage'
import AssistantMessage from './AssistantMessage'

/** Peer buttons in the same horizontal group as the given control. */
function rowButtonCount(el: HTMLElement): number {
  const row = el.closest('[data-message-actions]') ?? el.parentElement!
  return row.querySelectorAll('button').length
}

describe('the thread action on a user message', () => {
  const props = {
    content: 'Nine issues came in overnight.',
    meta: { mid: 'm-1' },
    messageTs: '2026-09-29T07:02:00Z',
    slotKey: 'chat-41',
    renderContent: (t: string) => <span>{t}</span>,
  }

  it('is in the overflow menu, not a peer button in the row', () => {
    const onReplyInThread = vi.fn()
    renderWithProviders(<UserMessage {...props} onReplyInThread={onReplyInThread} onTogglePin={vi.fn()} />)
    const trigger = screen.getByTestId('user-message-more-actions')
    expect(trigger).toBeInTheDocument()
    fireEvent.click(trigger)
    const item = screen.getByTestId('reply-in-thread')
    expect(item.tagName).not.toBe('BUTTON')
    fireEvent.click(item)
    expect(onReplyInThread).toHaveBeenCalledTimes(1)
  })

  it('does not grow the row: the trigger stands where the pin button stood', () => {
    const withThread = renderWithProviders(
      <UserMessage {...props} onReplyInThread={vi.fn()} onTogglePin={vi.fn()} />,
    )
    const withCount = rowButtonCount(withThread.getByTestId('user-message-more-actions'))
    withThread.unmount()
    // The same row with no thread hooks at all: pin is a row button and there is
    // no trigger. Equal counts is the claim -- the rule is about growth.
    const plain = renderWithProviders(<UserMessage {...props} onTogglePin={vi.fn()} />)
    const plainRow = plain.container.querySelector('[data-message-actions]')!
    expect(plainRow.querySelectorAll('button').length).toBe(withCount)
  })
})

describe('the thread action on an assistant message', () => {
  const props = {
    content: 'Four are open.',
    text: 'Four are open.',
    meta: { mid: 'm-2' },
    messageTs: '2026-09-29T07:03:00Z',
    slotKey: 'chat-41',
    renderContent: (t: string) => <span>{t}</span>,
  }

  it('is in More on a finished reply', () => {
    const onReplyInThread = vi.fn()
    renderWithProviders(<AssistantMessage {...props} onReplyInThread={onReplyInThread} />)
    fireEvent.click(screen.getByTestId('assistant-more-actions'))
    const item = screen.getByTestId('reply-in-thread')
    expect(item.tagName).not.toBe('BUTTON')
    fireEvent.click(item)
    expect(onReplyInThread).toHaveBeenCalledTimes(1)
  })

  it('stays a visible control while the reply is still streaming', () => {
    // A streaming reply withholds its footer, so this row holds one control and
    // is under the cap on its own. Waiting for the turn to end to offer a thread
    // is the behaviour this change exists to remove.
    renderWithProviders(<AssistantMessage {...props} isStreaming onReplyInThread={vi.fn()} />)
    const btn = screen.getByTestId('reply-in-thread-streaming')
    expect(btn.tagName).toBe('BUTTON')
    expect(rowButtonCount(btn)).toBe(1)
  })
})
