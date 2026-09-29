/**
 * Screenshot harness for the CREW LOG side-panel section.
 *
 * Three scenarios against the REAL built SPA (website/dist), gateway-free — the
 * five projection routes are answered from fixtures, so the same bytes render on
 * every run:
 *
 *  1. crew-log-dark.png  — the section on a session with a populated crew log,
 *     Status and Usage open (their default), the three list folds collapsed but
 *     carrying their counts in the header.
 *  2. crew-log-light.png — the same state in the light theme, which is what
 *     proves the section is drawn from theme variables rather than fixed colours.
 *  3. crew-log-empty.png — a session whose folds are all at seq 0: the section
 *     says nothing is recorded instead of rendering five zeroed sections.
 *  4. crew-log-off.png — the same empty folds with the gateway reporting recording
 *     switched off: the section says so and names the flag, which the empty frame
 *     must not.
 *
 * Each scenario ASSERTS before it photographs, so the run exits non-zero when the
 * section is absent or renders the wrong state (a harness that only writes a PNG
 * fails toward a green run with a blank picture).
 *
 * Usage: node scripts/capture-crew-log-panel.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/crew-log'

mkdirSync(OUT, { recursive: true })

const SLOT = 'chat-crew-log'
const now = Math.floor(Date.now() / 1000)

const slots = [{
  key: SLOT, title: 'Crew log demo', running: false, last_message: '', messages: 8,
  agent: 'kirocrew', memory_mode: 'persistent', project: '', folder_id: '',
  modified: now, tags: [], source_links: [], source_links_total: 0,
}]

/** A fold's envelope, exactly as `api_session_crew_log_projection` answers. */
const fold = (name, seq, value) => ({ session_id: SLOT, name, seq, value })

const POPULATED = {
  status: fold('status', 1842, {
    lifecycle: 'open',
    opened_at: 1789821630000,
    closed_at: null,
    close_reason: null,
    resumed: false,
    seeded: true,
    agent: 'kirocrew',
    owner: 'owner',
    slot: SLOT,
    cwd: '',
    model: 'a-model',
    provider: 'acp',
    turn: null,
    turn_open: false,
    turns_completed: 12,
    turns_refused: 1,
    last_stop_reason: 'end_turn',
    last_error: '',
    last_time: 1789822140000,
    entries: 1842,
    dropped: { count: 0, bytes: 0 },
  }),
  usage: fold('usage', 1842, {
    turns: { completed: 12, credits_reported: 12, tokens_reported: 12, duration_reported: 12 },
    credits: 8.41,
    tokens: { input: 918220, output: 96431, cache_read: 182004, cache_write: 8258, total: 1204913 },
    duration_ms: 1624000,
    by_model: { 'a-model': { turns: 12, credits: 8.41, credits_reported: 12, tokens: 1204913 } },
    models_omitted: 0,
    models_omitted_saturated: false,
    context: { tokens: 42000, chars: 168000, blocks: 9, estimated_turns: 0, by_source: {} },
    compactions: { count: 3, freed_pct: 41 },
    steps: { completed: 40, ms: 900 },
  }),
  timeline: fold('timeline', 1842, {
    moments: [
      { seq: 1, time: 1789821630000, type: 'session/opened' },
      { seq: 4, time: 1789821750000, type: 'model/selected' },
      { seq: 1409, time: 1789821810000, type: 'turn/refused' },
      { seq: 1540, time: 1789821870000, type: 'turn/completed' },
      { seq: 1702, time: 1789821930000, type: 'compaction/applied' },
      { seq: 1834, time: 1789821990000, type: 'turn/completed' },
      { seq: 1838, time: 1789822020000, type: 'turn/started' },
    ],
    dropped: 14,
    limit: 200,
    first_seq: 1,
    last_seq: 1838,
  }),
  tools: fold('tools', 1840, {
    calls: 96,
    completed: 94,
    errors: 4,
    open: 2,
    open_calls: [
      { call_id: 'c-1', name: 'execute_bash', time: 1789822050000, seq: 1839, turn: 12 },
      { call_id: 'c-2', name: 'fs_read', time: 1789822110000, seq: 1840, turn: 12 },
    ],
    open_calls_omitted: 0,
    open_dropped: 0,
    unidentified_calls: 0,
    unmatched_completions: 0,
    elapsed_ms: 74434,
    by_name: {
      fs_read: { calls: 41, completed: 41, errors: 0, elapsed_ms: 9120, last_status: 'ok', last_time: '', servers: [], servers_omitted: 0, servers_omitted_saturated: false },
      execute_bash: { calls: 28, completed: 25, errors: 3, elapsed_ms: 61400, last_status: 'error', last_time: '', servers: [], servers_omitted: 0, servers_omitted_saturated: false },
      fs_write: { calls: 17, completed: 17, errors: 0, elapsed_ms: 2044, last_status: 'ok', last_time: '', servers: [], servers_omitted: 0, servers_omitted_saturated: false },
      grep: { calls: 10, completed: 9, errors: 1, elapsed_ms: 1870, last_status: 'ok', last_time: '', servers: [], servers_omitted: 0, servers_omitted_saturated: false },
    },
    names_omitted: 0,
    names_omitted_saturated: false,
  }),
  approvals: fold('approvals', 1842, {
    requested: 4,
    decided: 3,
    pending: 1,
    pending_requests: [
      { approval_id: 'a-4', tool: 'execute_bash', reason: '', turn: 12, time: 1789822140000, seq: 1841 },
    ],
    pending_omitted: 0,
    pending_dropped: 0,
    unidentified_requests: 0,
    unmatched_decisions: 0,
    by_decision: { approved: 3 },
    last: { approval_id: 'a-3', decision: 'approved', by: 'owner', cause: '', tool: 'fs_write', turn: 11, time: 1789821900000, seq: 1700 },
  }),
  // Three children, one per state the credits and elapsed cells have to tell apart:
  // finished having reported a charge, closed having reported none, and still running.
  // A fixture with only the first would photograph the one case that cannot go wrong.
  subagents: fold('subagents', 1842, {
    by_id: {
      'sub-1': {
        agent_id: 'sub-1', seq_spawned: 1204, turn: 9,
        agent: 'kirocrew-worker', model: 'a-model',
        outcome: 'completed', ms: 412000, credits: 3.18, reason: '', steers: 2,
      },
      'sub-2': {
        agent_id: 'sub-2', seq_spawned: 1388, turn: 11,
        agent: 'kirocrew-lite', model: 'another-model',
        outcome: 'stopped', ms: 9400, credits: null, reason: 'the owner stopped it', steers: 0,
      },
      'sub-3': {
        agent_id: 'sub-3', seq_spawned: 1836, turn: 12,
        agent: 'kirocrew-worker', model: 'a-model',
        outcome: null, ms: 0, credits: null, reason: '', steers: 1,
      },
    },
    open: ['sub-3'],
    running: 1,
    omitted: 0,
    limit: 512,
    totals: {
      spawned: 3, completed: 1, failed: 0, stopped: 1, unknown: 0,
      closed_unmatched: 0, steers: 3, ms: 421400, credits: 3.18, credits_reported: 1,
    },
  }),
}

/** Every fold at seq 0 — what a session with no crew log reads back. */
const EMPTY = Object.fromEntries(
  Object.keys(POPULATED).map(name => [name, fold(name, 0, {})]),
)

/** The seeded panel bucket: the crew-log tab present and focused, so the capture
 *  lands on the section without driving the + menu. The bucket shape is
 *  `usePanelTabs`'s own (`mc-panel-tabs:<slot>`). */
const PANEL_BUCKET = JSON.stringify({
  activeId: 'crewlog',
  tabs: [{ id: 'crewlog', kind: 'crewlog', title: 'Crew log' }],
})

async function renderPanel(
  browser, base, {
    theme, bundle, open, close, resolved = true, drained = true, recording = true,
    flagValue = 'fasle',
  },
) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    slots,
    theme,
    localStorageEntries: {
      // The section is a diagnostics view, so its + menu row and its tab live
      // behind the Developer Mode consent gate.
      'mc-dev-mode': '1',
      ['mc-activity-open:' + SLOT]: 'true',
      ['mc-panel-tabs:' + SLOT]: PANEL_BUCKET,
      'mc-side-panel-width': '430',
    },
    extra: async (path, route) => {
      // The panel reads all five folds in ONE request, so the stub answers the
      // batch route the client actually calls -- a per-name stub would leave the
      // real client unexercised and the panel empty.
      if (/^\/api\/sessions\/[^/]+\/crew-log\/projections$/.test(path)) {
        // The two flags are sent EXPLICITLY, not left to the client's defaults: a
        // frame is evidence of what the reader sees, and the footer and the empty
        // state both change wording on these. A capture that omitted them would
        // photograph the default branch while claiming to show the real one.
        await json(route, {
          session_id: SLOT, projections: bundle, resolved, writes_drained: drained, recording,
          ...(recording ? {} : {
            flag_value: flagValue,
            flag_recognised: ['0', 'false', 'no', 'off'].includes(flagValue.toLowerCase()),
            env_file: '~/.kiro/crew/.env',
          }),
        })
        return true
      }
      return false
    },
  })
  await page.goto(`${base}/chat?slot=${encodeURIComponent(SLOT)}`)
  await page.waitForSelector('[data-testid="crew-log-tab"]', { timeout: 15000 })
  // The theme SEED is not the theme RENDERED: the shell resolves its palette from
  // the boot payload, so a capture that only sets localStorage can photograph the
  // wrong mode and still exit 0 — which is the whole point of a light/dark pair.
  const rendered = await page.evaluate(() => ({
    theme: document.documentElement.getAttribute('data-theme') || '',
    bg: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),
  }))
  if (!rendered.theme.includes(theme)) {
    throw new Error(`${theme}: shell rendered ${JSON.stringify(rendered)} instead`)
  }
  // Only Status and Usage open themselves, so any other body has to be clicked
  // open before it can be photographed -- a collapsed section is not evidence of
  // what it holds.
  for (const heading of [...(open ?? []), ...(close ?? [])]) {
    await page.getByText(heading, { exact: true }).click()
    await page.waitForTimeout(150)
  }
  await page.waitForTimeout(500)
  return { context, page, rendered }
}

async function sectionText(page) {
  return page.locator('[data-testid="crew-log-tab"]').innerText()
}

function assertContains(scenario, text, expected) {
  for (const needle of expected) {
    if (!text.includes(needle)) {
      throw new Error(`${scenario}: section does not show ${JSON.stringify(needle)}`)
    }
  }
}

/** The panel COLUMN, strip included.
 *
 *  Not the section body alone: a frame cropped to the body shows no tab, so
 *  nothing in the picture says which panel it is -- which is exactly what a blind
 *  reader reported. The strip carries the "Crew log" tab and its glyph, so one
 *  frame answers both "what is this" and "how did you get here". */
async function shootPanel(page, name) {
  const strip = page.locator('[data-testid="crew-log-tab"]').locator('xpath=ancestor::*[.//button[@title="Open side panel tab"]][1]')
  const target = (await strip.count()) ? strip.first() : page.locator('[data-testid="crew-log-tab"]')
  await target.screenshot({ path: `${OUT}/${name}.png` })
  console.log(`wrote ${OUT}/${name}.png`)
}

/** The + menu open on its diagnostics group, so the row that reaches this view is
 *  in the evidence rather than described in prose. */
async function shootMenu(page, name) {
  await page.locator('button[title="Open side panel tab"]').first().click()
  await page.getByRole('menu').waitFor({ timeout: 5000 })
  // The menu fades and scales in, so a frame taken on the first paint catches it
  // semi-transparent -- which reads as a rendering fault rather than a menu.
  await page.waitForTimeout(450)
  const row = page.getByText('What this session did and what it cost', { exact: false })
  if (!(await row.count())) {
    const alt = page.getByRole('menuitem').filter({ hasText: 'Crew log' })
    if (!(await alt.count())) throw new Error('menu: no Crew log row -- is Developer Mode seeded?')
  }
  await page.screenshot({ path: `${OUT}/${name}.png` })
  console.log(`wrote ${OUT}/${name}.png`)
  await page.keyboard.press('Escape')
}

async function main() {
  const { srv, base } = await serveDist()
  // chromiumSandbox:false — the CI and dev hosts here cannot run Chromium's
  // sandbox, the same reason every sibling harness passes it.
  const browser = await chromium.launch({ chromiumSandbox: false })
  try {
    for (const theme of ['dark', 'light']) {
      const { context, page, rendered } = await renderPanel(browser, base, {
        theme, bundle: POPULATED, open: theme === 'light' ? ['Timeline'] : [],
      })
      console.log(`${theme}: data-theme=${rendered.theme} --bg=${rendered.bg}`)
      const text = await sectionText(page)
      assertContains(`populated/${theme}`, text, [
        'Status', 'Usage', 'Timeline', 'Tools', 'Approvals', 'Subagents',
        // The count line names no actor: the reader of a tile has nowhere to learn
        // what the gateway is, and the empty state is where that word is earned.
        'completed: 12 · refused: 1',
        'calls: 96 · unfinished: 2',
        'asked: 4 · pending: 1',
        'dispatched: 3 · running: 1',
        'up to date through entry 1,842',
      ])
      if (theme === 'light') {
        // The light shot is taken with a list fold opened, so the pair of
        // screenshots covers both a collapsed and an expanded body.
        assertContains('populated/light', text, [
          'turn started', 'session opened', 'context compacted',
          // A moment's number is the entry it was recorded AT, which is not the
          // footer's watermark -- the heading has to say so, or the two numbers
          // read as one counter disagreeing with itself. Asserted in caps because
          // the column heading is CSS-uppercased, so that is what a reader sees.
          'AT ENTRY', 'turn blocked by a gate',
        ])
      }
      await shootPanel(page, `crew-log-${theme}`)
      if (theme === 'dark') await shootMenu(page, 'crew-log-menu')
      await context.close()
    }

    {
      // Tools and Approvals collapsed in every other frame, so their bodies went
      // unseen: the per-tool table, the unfinished calls, the rows waiting on the
      // reader, and the line telling the reader where to answer them.
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: POPULATED,
        open: ['Tools', 'Approvals'],
        // Status and Usage are the two that open themselves, and together they are
        // taller than the column: left open, they push the approval rows below the
        // fold, so the frame would again show everything except the thing asked
        // about.
        close: ['Status', 'Usage'],
      })
      const text = await sectionText(page)
      assertContains('folds/dark', text, [
        // The stat and the column share one word now: they count the same thing,
        // and "failed calls" beside a column headed "errors" read as two concepts.
        'Unfinished', 'execute_bash', 'fs_read', 'tool calls', 'errors',
        'Waiting for you', 'Answer these on the approval card in the conversation.',
      ])
      await shootPanel(page, 'crew-log-folds')
      await context.close()
    }

    {
      // The subagents table, which no other frame opens. What it has to prove is the
      // credits and elapsed cells telling THREE states apart -- a reported charge, a
      // child that reported none, and a child still running -- because a zero in
      // either column would present the absence of a measurement as a measurement.
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: POPULATED,
        open: ['Subagents'],
        // Status and Usage open themselves and together are taller than the column,
        // so the table would sit below the fold in the frame that is about it.
        close: ['Status', 'Usage'],
      })
      const text = await sectionText(page)
      assertContains('subagents/dark', text, [
        'subagents dispatched', 'credits, all subagents',
        'agent', 'model', 'outcome', 'elapsed', 'credits',
        'kirocrew-worker', 'kirocrew-lite', 'another-model',
        'finished', 'stopped', 'running',
        // The middle state, which is the one a panel gets wrong.
        'not said',
      ])
      // CHILD rows only: a child that did not finish also emits a full-width row carrying
      // its closer's reason, which has one cell rather than five.
      const { rows, reasons } = await page.evaluate(() => {
        const table = [...document.querySelectorAll('[data-testid="crew-log-tab"] table')]
          .find(t => t.textContent?.includes('outcome'))
        const all = [...(table?.querySelectorAll('tbody tr') ?? [])].map(tr =>
          [...tr.querySelectorAll('td')].map(td => td.textContent?.trim() ?? ''))
        return { rows: all.filter(r => r.length === 5), reasons: all.filter(r => r.length === 1) }
      })
      if (rows.length !== 3) {
        throw new Error(`subagents: ${rows.length} row(s) drawn, expected 3`)
      }
      // The reason row is what gives the retained `reason` field a readable consumer.
      if (!reasons.some(r => r[0].includes('the owner stopped it'))) {
        throw new Error(`subagents: no reason row drawn: ${JSON.stringify(reasons)}`)
      }
      // Dispatch order, which is the order the fold renders and the order a reader of a
      // session's history expects. Matched with startsWith, not equality: the agent cell
      // carries a second line (the turn, the steer count, a failure's reason), so its full
      // text is the name followed by that detail.
      if (!rows[0][0].startsWith('kirocrew-worker') || !rows[1][0].startsWith('kirocrew-lite')) {
        throw new Error(`subagents: rows out of dispatch order: ${JSON.stringify(rows)}`)
      }
      // The detail line is what gives the retained turn / steers / reason fields a reader.
      if (!rows[0][0].includes('turn 9') || !rows[0][0].includes('corrections: 2')) {
        throw new Error(`subagents: row detail missing turn/steers: ${JSON.stringify(rows[0])}`)
      }

      // The running child: neither a charge nor a duration is known yet, and drawing
      // 0 for either is the defect this assertion exists for.
      if (rows[2][3] !== '—' || rows[2][4] !== 'not said') {
        throw new Error(`subagents: running child drew ${JSON.stringify(rows[2])}`)
      }
      if (rows[1][4] !== 'not said') {
        throw new Error(`subagents: an unreported charge drew ${JSON.stringify(rows[1])}`)
      }
      if (rows[0][4] === 'not said') {
        throw new Error(`subagents: a REPORTED charge drew "not said": ${JSON.stringify(rows[0])}`)
      }
      await shootPanel(page, 'crew-log-subagents')
      await context.close()
    }

    {
      // The subagents fold past its retention cap, plus a closer that matched no
      // dispatch. Both lines exist so the totals exceeding what the table accounts
      // for reads as the bound speaking rather than as an arithmetic bug.
      const CAPPED = structuredClone(POPULATED)
      Object.assign(CAPPED.subagents.value, { omitted: 9, running: 12 })
      Object.assign(CAPPED.subagents.value.totals, { spawned: 14, closed_unmatched: 2 })
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: CAPPED, open: ['Subagents'], close: ['Status', 'Usage'],
      })
      const text = await sectionText(page)
      assertContains('subagents-capped/dark', text, [
        'subagents not detailed: 9',
        '2 more finished, but this list does not have their start',
        // The header counts every dispatch, not the three rows below it.
        'dispatched: 14 · running: 12',
      ])
      await shootPanel(page, 'crew-log-subagents-capped')
      await context.close()
    }

    {
      // A slot whose ACP session was torn down: the record is on disk under the
      // retired id, so this state exists precisely to NOT say "nothing recorded".
      // It is the peer of the empty frame and carries the heavier copy of the two.
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: EMPTY, resolved: false,
      })
      const text = await sectionText(page)
      assertContains('unaddressable', text, [
        'Nothing saved since this chat’s last reset',
        // Each empty state opens with the ACTION that differs between them, which
        // is what a reader uses to tell the pair apart; `resolved: false` covers a
        // fresh slot too, so the retired case stays conditional.
        'Run a turn to start saving messages again',
        'has not run a turn yet',
        'Messages saved before the reset stay saved',
      ])
      if (text.includes('No messages saved for this chat yet')) {
        throw new Error('unaddressable: claimed nothing was recorded for a torn-down session')
      }
      if (text.includes('no entries yet')) {
        throw new Error('unaddressable: the footer claims an up-to-date record that does not exist')
      }
      await shootPanel(page, 'crew-log-unaddressable')
      await context.close()
    }

    {
      // The branches a single populated frame cannot show at once: the writer still
      // owing entries, more than one model (the per-model table renders only then),
      // a turn in flight, entries lost to retention, and both truncation lines.
      const VARIANTS = structuredClone(POPULATED)
      Object.assign(VARIANTS.usage.value, {
        by_model: {
          'a-model': { turns: 8, credits: 5.2, credits_reported: 8, tokens: 800000 },
          'another-model': { turns: 4, credits: 3.21, credits_reported: 4, tokens: 404913 },
        },
        models_omitted: 2,
      })
      Object.assign(VARIANTS.status.value, {
        turn_open: true,
        turn: { turn: 13 },
        dropped: { count: 14, bytes: 9000 },
      })
      Object.assign(VARIANTS.tools.value, { names_omitted: 6 })
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: VARIANTS, drained: false, close: ['Timeline'],
      })
      const text = await sectionText(page)
      assertContains('variants/dark', text, [
        'writes still pending',
        'another-model',
        'models not detailed: 2',
        'turn 13',
        'Removed by retention',
      ])
      await shootPanel(page, 'crew-log-variants')
      await context.close()
    }

    {
      const { context, page } = await renderPanel(browser, base, { theme: 'dark', bundle: EMPTY })
      const text = await sectionText(page)
      assertContains('empty', text, [
        'No messages saved for this chat yet', 'up to date; no entries yet',
        'a chat that has just started has none saved yet',
      ])
      // Recording is on here, so the switch-off instructions belong to the other frame.
      if (text.includes('KIROCREW_CREW_LOG')) {
        throw new Error('empty: recording is on but the switch-off instructions are shown')
      }
      if (text.includes('Lifecycle')) {
        throw new Error('empty: the section rendered fold bodies for a log with no entries')
      }
      await shootPanel(page, 'crew-log-empty')
      await context.close()
    }

    {
      // The gateway reports recording switched off: the one empty state whose fix is
      // the flag rather than a turn.
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: EMPTY, recording: false,
      })
      const text = await sectionText(page)
      assertContains('off', text, [
        'The crew log is off',
        'Messages in this chat are not being saved.',
        'KIROCREW_CREW_LOG has a value that is not recognised, “fasle”',
        'Crew log off',
        'Remove that setting',
        'what the crew log stores',
      ])
      if (text.includes('No messages saved for this chat yet')) {
        throw new Error('off: said nothing was recorded instead of that recording is off')
      }
      const panelText = await page.locator('[data-testid="crew-log-tab"]').innerText()
      if (panelText.includes('this chat is writing now') || panelText.includes('no entries yet')) {
        throw new Error('off: the footer still describes a record while recording is off')
      }
      if (!(await page.getByRole('link', { name: 'what the crew log stores' }).count())) {
        throw new Error('off: the docs path is not a link')
      }
      const notice = page.getByTestId('crew-log-off')
      if (!(await notice.count()) || !(await notice.getAttribute('class')).includes('bg-warn-subtle')) {
        throw new Error('off: rendered like the neutral empty state instead of a warning')
      }
      await shootPanel(page, 'crew-log-off')
      await context.close()
    }

    {
      // Recording switched off on a chat that already has entries: the warning sits
      // above the folds, which stay readable; the footer leads with the status beside
      // their watermark, and the scope note about the log being written now is gone.
      const { context, page } = await renderPanel(browser, base, {
        theme: 'dark', bundle: POPULATED, recording: false, flagValue: '0',
      })
      const text = await sectionText(page)
      assertContains('off-entries', text, [
        'The crew log is off',
        'New messages in this chat are not being saved. The ones below stay.',
        'Remove KIROCREW_CREW_LOG from ~/.kiro/crew/.env',
        'It is set to “0”.',
        'Status', 'Usage', 'Timeline', 'Tools', 'Approvals',
        'Crew log off · last saved entry 1,842',
      ])
      if (text.includes('not recognised')) {
        throw new Error('off-entries: a switch-off spelling was worded as unrecognised')
      }
      if (text.includes('this chat is writing now')) {
        throw new Error('off-entries: the scope note still claims a log is being written')
      }
      const notice = page.getByTestId('crew-log-off')
      if (!(await notice.count()) || !(await notice.getAttribute('class')).includes('bg-warn-subtle')) {
        throw new Error('off-entries: no warning notice above the saved entries')
      }
      const above = await page.evaluate(() => {
        const n = document.querySelector('[data-testid="crew-log-off"]')
        const status = [...document.querySelectorAll('[data-testid="crew-log-tab"] *')]
          .find(el => el.textContent?.trim() === 'Status')
        return Boolean(n && status && (n.compareDocumentPosition(status) & Node.DOCUMENT_POSITION_FOLLOWING))
      })
      if (!above) throw new Error('off-entries: the warning is not above the fold tables')
      await shootPanel(page, 'crew-log-off-entries')
      await context.close()
    }
  } finally {
    await browser.close()
    srv.close()
  }
}

await main()
