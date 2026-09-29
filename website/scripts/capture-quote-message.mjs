/** Real-browser evidence for quoting a whole message.
 *
 * Drives website/capture/quote-message.html (real UserMessage / AssistantMessage /
 * ChatInput + the real useMessageQuote hook). Per theme, and asserted:
 *
 *   1-row       hover on Kiro's reply: the row shows Quote + More (two seats)
 *   2-menu      right-click on the bubble: the context menu, Quote first
 *   3-composer  the quote staged INSIDE the input box as a card, with Remove
 *   4-sent      Send: the new user row draws the quote card, body without `>`
 *   5-narrow    390px phone frame: long-press menu open over the staged card
 *   6-unstaged  Remove on the card clears it (composer back to plain)
 *   7-crewmate-menu   the crewmate DM (real ChatMessageList + crewmate renderers,
 *                     Reply in thread in the row): the bubble menu, Quote first
 *   8-crewmate-more   the crewmate reply's More menu: Quote first, then Copy...
 *   9-crewmate-sent   a DM row carrying the quote card
 *
 * Usage:
 *   npx vite --host 127.0.0.1 --port 6872 --strictPort     # in website/
 *   node scripts/capture-quote-message.mjs http://127.0.0.1:6872 ../temp-screenshots/quote-message
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

const BASE = process.argv[2] || 'http://127.0.0.1:6872'
const OUT = resolve(process.argv[3] || '../temp-screenshots/quote-message')
mkdirSync(OUT, { recursive: true })
const { LD_LIBRARY_PATH: _mise, ...browserEnv } = process.env
const browser = await chromium.launch({ env: browserEnv })
let failures = 0
const check = (l, ok) => { console.log(`${l} => ${ok ? 'OK' : 'FAIL'}`); if (!ok) failures++ }

async function open(theme, scene, narrow = false, extra = '') {
  const ctx = await browser.newContext({ viewport: narrow ? { width: 390, height: 780 } : { width: 1100, height: 720 }, deviceScaleFactor: 2, hasTouch: narrow, isMobile: narrow })
  const page = await ctx.newPage()
  const errors = []; page.on('pageerror', e => errors.push(String(e)))
  await page.goto(`${BASE}/capture/quote-message.html?theme=${theme}&scene=${scene}${extra}`, { waitUntil: 'networkidle' })
  await page.waitForSelector('textarea'); await page.waitForTimeout(400)
  return { ctx, page, errors }
}
const shot = (page, name) => page.screenshot({ path: resolve(OUT, name) })

for (const theme of ['dark', 'light']) {
  // 1 row
  {
    const { ctx, page, errors } = await open(theme, 'hover')
    const kiro = page.locator('[data-role="assistant"]').first()
    await kiro.hover({ position: { x: 120, y: 30 } }); await page.waitForTimeout(800)
    const row = kiro.locator('[data-testid="quote-message"]').locator('..')
    const labels = await row.locator('> button').evaluateAll(bs => bs.map(b => b.getAttribute('aria-label')))
    check(`[${theme}/row] Kiro row = Quote + More`, JSON.stringify(labels) === JSON.stringify(['Quote message', 'More actions']))
    await shot(page, `${theme}-1-row.png`)
    check(`[${theme}/row] no page errors`, errors.length === 0); await ctx.close()
  }
  // 2 menu
  {
    const { ctx, page, errors } = await open(theme, 'hover')
    const kiro = page.locator('[data-role="assistant"]').first()
    await kiro.locator('[data-testid="message-bubble"]').click({ button: 'right', position: { x: 140, y: 30 } })
    await page.waitForSelector('[data-testid="message-context-menu"]'); await page.waitForTimeout(200)
    const items = await page.getByRole('menuitem').allInnerTexts()
    check(`[${theme}/menu] Quote first, then Copy / Copy link / Pin`, JSON.stringify(items) === JSON.stringify(['Quote message', 'Copy', 'Copy link to message', 'Pin message']))
    await shot(page, `${theme}-2-menu.png`)
    // Selecting Quote stages the card in the composer.
    await page.getByTestId('message-context-quote').click()
    await page.waitForSelector('[data-testid="quote-card-composer"]')
    check(`[${theme}/menu] Quote from the menu stages the card`, await page.getByTestId('quote-card-composer').isVisible())
    check(`[${theme}/menu] no page errors`, errors.length === 0); await ctx.close()
  }
  // 3 composer (+ 6 unstaged)
  {
    const { ctx, page, errors } = await open(theme, 'composer')
    const card = page.getByTestId('quote-card-composer')
    const inside = await card.evaluate(el => !!el.closest('[data-testid="input-wrapper"]'))
    check(`[${theme}/composer] card is inside the input box`, inside)
    check(`[${theme}/composer] card names the author + excerpt`, (await card.innerText()).includes('Quoting Assistant') && (await card.innerText()).includes('Three things landed'))
    await shot(page, `${theme}-3-composer.png`)
    await page.getByTestId('quote-card-remove').click(); await page.waitForTimeout(150)
    check(`[${theme}/composer] Remove clears the card, draft kept`, (await card.count()) === 0 && (await page.locator('textarea[data-composer-input]').inputValue()).startsWith('Can the chips'))
    await shot(page, `${theme}-6-unstaged.png`)
    check(`[${theme}/composer] no page errors`, errors.length === 0); await ctx.close()
  }
  // 4 sent
  {
    const { ctx, page, errors } = await open(theme, 'sent')
    await page.waitForSelector('[data-testid="quote-card-sent"]')
    const sentRow = page.locator('[data-role="user"]').nth(1)
    const bubbleText = await sentRow.locator('.message-bubble').innerText()
    check(`[${theme}/sent] card + body, no raw '>' lines`, bubbleText.includes('Assistant') && bubbleText.includes('Can the chips get a fixed height') && !bubbleText.includes('> '))
    check(`[${theme}/sent] card is the jump control`, await sentRow.getByRole('button', { name: 'Jump to the quoted message' }).isVisible())
    await sentRow.getByRole('button', { name: 'Jump to the quoted message' }).click()
    check(`[${theme}/sent] jump hands over the quoted ts`, (await page.locator('[data-capture-root]').getAttribute('data-jumped')) === '2026-09-29T09:12:40Z')
    await page.mouse.move(5, 5); await page.waitForTimeout(400)
    await shot(page, `${theme}-4-sent.png`)
    check(`[${theme}/sent] no page errors`, errors.length === 0); await ctx.close()
  }
  // 5 narrow
  {
    const { ctx, page, errors } = await open(theme, 'narrow', true)
    await page.locator('[data-role="assistant"]').first().locator('[data-testid="message-bubble"]').click({ button: 'right', position: { x: 120, y: 30 } })
    await page.waitForSelector('[data-testid="message-context-menu"]'); await page.waitForTimeout(200)
    check(`[${theme}/narrow] no sideways scroll`, (await page.evaluate(() => document.documentElement.scrollWidth)) <= 390)
    await shot(page, `${theme}-5-narrow.png`)
    check(`[${theme}/narrow] no page errors`, errors.length === 0); await ctx.close()
  }
  // 7-9 crewmate DM
  {
    const { ctx, page, errors } = await open(theme, 'hover', false, '&host=crewmate')
    await page.waitForSelector('[data-testid="crewmate-message"]')
    const bubble = page.locator('[data-testid="crewmate-message"] [data-testid="message-bubble"]').first()
    await bubble.click({ button: 'right', position: { x: 140, y: 30 } })
    await page.waitForSelector('[data-testid="message-context-menu"]'); await page.waitForTimeout(200)
    check(`[${theme}/crewmate] bubble menu, Quote first`, (await page.getByRole('menuitem').first().innerText()).includes('Quote message'))
    await shot(page, `${theme}-7-crewmate-menu.png`)
    await page.keyboard.press('Escape'); await page.waitForTimeout(150)
    const row = page.locator('[data-testid="crewmate-message"]').first()
    await row.hover({ position: { x: 120, y: 30 } }); await page.waitForTimeout(700)
    const seats = await row.locator('[data-testid="reply-in-thread"]').locator('..').locator('> button').evaluateAll(bs => bs.map(b => b.getAttribute('aria-label')))
    check(`[${theme}/crewmate] row keeps Reply + More`, JSON.stringify(seats) === JSON.stringify(['Reply in thread', 'More actions']))
    await row.getByTestId('assistant-more-actions').click(); await page.waitForTimeout(200)
    check(`[${theme}/crewmate] More: Quote first`, (await page.getByRole('menuitem').first().innerText()).includes('Quote message'))
    await shot(page, `${theme}-8-crewmate-more.png`)
    await page.getByTestId('quote-message-menu-item').click()
    await page.waitForSelector('[data-testid="quote-card-composer"]')
    check(`[${theme}/crewmate] Quote from More stages the card`, await page.getByTestId('quote-card-composer').isVisible())
    check(`[${theme}/crewmate] no page errors`, errors.length === 0); await ctx.close()
  }
  {
    const { ctx, page, errors } = await open(theme, 'sent', false, '&host=crewmate')
    await page.waitForSelector('[data-testid="quote-card-sent"]')
    const text = await page.locator('[data-testid="quote-card-sent"]').locator('..').innerText()
    check(`[${theme}/crewmate-sent] card + body, no raw '>'`, text.includes('Can the chips get a fixed height') && !text.includes('> '))
    await page.mouse.move(5, 5); await page.waitForTimeout(300)
    await shot(page, `${theme}-9-crewmate-sent.png`)
    check(`[${theme}/crewmate-sent] no page errors`, errors.length === 0); await ctx.close()
  }
}
await browser.close()
console.log(failures ? `${failures} check(s) FAILED` : 'all checks OK')
process.exit(failures ? 1 : 0)
