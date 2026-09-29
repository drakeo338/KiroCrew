// Evidence capture: the composer while an automatic compaction holds the session,
// and the transcript card a declined Stop settles into.
//
// Two shots against the REAL built SPA (website/dist) with the dashboard API
// stubbed. Not a test: the vitest files pin the behaviour; this shows it.
//
//   node scripts/capture-compacting-stop.mjs [outDir]
import { chromium } from 'playwright'
import { mkdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/compacting-stop'
mkdirSync(OUT, { recursive: true })

const LOCALES = fileURLToPath(new URL('../src/i18n/locales/', import.meta.url))
const manual = JSON.parse(readFileSync(LOCALES + 'en.manual.json', 'utf-8'))
const HINT = manual.components.chatInput.compacting_context
const CARD = manual.pages.chat.stopEventCard.stop_declined_compacting_2
if (!HINT || !CARD) throw new Error('compacting keys missing from en.manual.json')

// The two gateway notices this change words (dashboard/state.py), verbatim, so
// the shot shows the text the CompactionCard renders for them.
const CANCELLED_NOTICE =
  '\u23F9 Stop ended the compaction that started at 87% of the context limit. ' +
  'The session kept running as it was; it will try to compact again later, or ' +
  'run `/compact` to do it now.'
const RESTART_NOTICE =
  '\u267B\uFE0F Compaction didn\'t succeed at 91%, so the session was restarted instead. ' +
  "The conversation above is still here, and the agent's next reply starts from a " +
  'recent excerpt of it rather than the whole thing.'

const SLOT = 'compacting-demo'
const now = Math.floor(Date.now() / 1000)

// The gateway writes the envelope into `cls`, mirrors it into `content`, and
// serves it parsed as `meta` (dashboard/state.py parse_cls_meta). All three, as
// the HTTP history endpoint would.
const stopMeta = {
  kind: 'stop_event',
  id: 'stop-demo',
  state: 'stop_declined_compacting',
  outcome: 'compacting',
  ts_start: new Date((now - 5) * 1000).toISOString(),
  ts_end: new Date((now - 4) * 1000).toISOString(),
}
const stopCard = JSON.stringify(stopMeta)

const slots = [{
  key: SLOT,
  title: 'Long investigation session',
  running: false,
  compacting: true,
  stop_state: 'idle',
  last_message: 'Cross-referenced the last three incident reports.',
  messages: 7,
  agent: 'kirocrew',
  memory_mode: 'persistent',
  modified: now,
  source_links: [],
  source_links_total: 0,
}]

const detail = {
  running: false,
  has_more: false,
  total: 7,
  queue: [],
  messages: [
    { role: 'user', ts: now - 900, content: 'Keep going through the incident reports and summarise each one.' },
    { role: 'assistant', ts: now - 60, content: 'Cross-referenced the last three incident reports. Two share a root cause in the retry path.' },
    { role: 'assistant', ts: now - 5, content: CANCELLED_NOTICE, cls: 'msg msg-a', kind: 'compaction', meta: { kind: 'compaction' } },
    { role: 'user', ts: now - 4, content: 'Carry on with the fourth report.' },
    { role: 'assistant', ts: now - 3, content: 'Fourth report: the retry path again, same root cause as the first two.' },
    { role: 'assistant', ts: now - 2, content: RESTART_NOTICE, cls: 'msg msg-a', kind: 'compaction', meta: { kind: 'compaction' } },
    // LAST, after the final assistant reply: the transcript folds the rows of a
    // finished turn under a "worked through N steps" disclosure, and a stop
    // card inside that fold is invisible in the full-page shot. Placed after
    // the reply it is a row of the CURRENT turn and renders open.
    { role: 'system', ts: now - 1, content: stopCard, cls: stopCard, meta: stopMeta },
  ],
  context_pct: 87,
  context_used_tokens: 174_000,
  context_window_tokens: 200_000,
}

async function main() {
  const { srv, base } = await serveDist()
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    slots,
    extra: async (path, route) => {
      if (path.startsWith('/api/chat/slots/')) { await json(route, detail); return true }
      return false
    },
  })
  await page.goto(base + `/chat?sid=${encodeURIComponent(SLOT)}`, { waitUntil: 'domcontentloaded' })
  await page.getByTestId('compacting-hint').waitFor({ timeout: 15000 })
  await page.getByText(HINT, { exact: true }).waitFor({ timeout: 5000 })
  await page.getByText(CARD, { exact: true }).waitFor({ timeout: 5000 })
  await page.getByText('kept running as it was', { exact: false }).first().waitFor({ timeout: 5000 })
  await page.getByText('recent excerpt', { exact: false }).first().waitFor({ timeout: 5000 })
  await page.waitForTimeout(300)
  await page.screenshot({ path: `${OUT}/compacting-composer.png` })
  const composer = page.getByTestId('compacting-indicator')
  await composer.screenshot({ path: `${OUT}/compacting-indicator.png` })
  const card = page.getByTestId('stop-event-card')
  await card.screenshot({ path: `${OUT}/stop-declined-card.png` })
  // The two notice rows, each as its own crop: the cancelled-by-Stop notice and
  // the reworded restart notice.
  const notices = page.locator('[data-testid="notice-card"], [data-testid="compaction-card"]')
  const count = await notices.count()
  if (count < 2) throw new Error(`expected 2 notice rows, found ${count}`)
  await notices.nth(count - 2).screenshot({ path: `${OUT}/cancelled-notice.png` })
  await notices.nth(count - 1).screenshot({ path: `${OUT}/restart-notice.png` })
  await browser.close()
  srv.close()
  console.log(`wrote ${OUT}/{compacting-composer,compacting-indicator,stop-declined-card,cancelled-notice,restart-notice}.png`)
}

main().catch(err => { console.error(err); process.exit(1) })
