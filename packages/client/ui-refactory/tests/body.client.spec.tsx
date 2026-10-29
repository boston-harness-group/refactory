// @vitest-environment jsdom
/**
 * The picker's body: what each read draws, and what each gesture sends.
 *
 * Props arrive the way the renderer composes them (a real store instance, a real
 * face over a scripted Remote), so a spec drives the component the way a person
 * does — choosing from a select, pressing a button — and reads the store and the
 * scripted Remote afterwards.
 */
import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, waitFor } from '@testing-library/react'
import { makeTranslate } from './translate.client.ts'
import { failureLine } from '../src/client/RefactoryBody.tsx'
import css from '../src/client/RefactoryBody.module.css'
import { en, zh } from '../src/client/locales.ts'
import { RENDER_WINDOW } from '../src/client/store.ts'
import type { RefactoryActivity } from '@deepseek-ai/dsh-refactory-console/types'
import {
  ACTIVITY, AUTHORING, CANDIDATES, COMPLETED, DISABLED, FRAME, GATED, GATED_STEERING, MODE, NO_STRATEGIES,
  PACKAGE, RECEIPT, SINGLE, STEERED, STEERING, STRATEGIES, ok,
} from './fake-console.client.ts'
import { TAB, mountBody } from './mount.client.tsx'
import { mockDialog } from './dialog.client.ts'

import { PROJECT } from './fake-console.client.ts'

const CHOICES = { project: '/repo', modes: [MODE, AUTHORING, DISABLED], strategies: STRATEGIES, packages: [PACKAGE], projects: [PROJECT] }

/** The knob values Start and Run once send before anything is changed. */
const KNOBS = ['rounds=3', 'temperature=0.5', 'source_root=/tmp/src', 'preset=broad', 'verbose=false']

let restoreDialog = () => {}
afterEach(() => { cleanup(); vi.restoreAllMocks(); restoreDialog() })

function openLedger(view: { container: HTMLElement }): void {
  restoreDialog = mockDialog()
  fireEvent.click(at(view, 'data-refactory-ledger-open'))
}

/** The one element a spec addresses by its data attribute. */
function at(view: { container: HTMLElement }, attribute: string, value?: string): HTMLElement {
  const selector = value === undefined ? `[${attribute}]` : `[${attribute}="${value}"]`
  const found = view.container.querySelector(selector)
  if (found === null) throw new Error(`no element for ${selector}`)
  return found as HTMLElement
}

/** The activity one mount hands the panel, with its journal paged. */
function live(after: number, limit: number, total: number): RefactoryActivity {
  return ACTIVITY(after, limit, total)
}

/** The `<li>` rows the activity section has in the DOM right now (T-18b). */
function rows(view: { container: HTMLElement }): Element[] {
  return [...view.container.querySelectorAll('[data-refactory-activity] li[data-refactory-activity-row]')]
}

/** Absent, as opposed to present and empty. */
function missing(view: { container: HTMLElement }, attribute: string, value?: string): boolean {
  const selector = value === undefined ? `[${attribute}]` : `[${attribute}="${value}"]`
  return view.container.querySelector(selector) === null
}

describe('failureLine', () => {
  it('names the three refusals the Host names precisely, in the reader\'s language', () => {
    const t = makeTranslate(zh)
    expect(failureLine(t, { code: 'refactory/unknown-mode', message: 'raw' })).toBe('这个模式不在目录里，请重新读取列表再选。')
    expect(failureLine(t, { code: 'refactory/authoring-mode', message: 'raw' })).toBe('这是创作模式，无法进行优化搜索；请使用「单次运行」。')
    expect(failureLine(t, { code: 'refactory/package-disabled', message: 'raw' })).toBe('这个模式的包已停用，请先启用它。')
  })

  it('shows the Host\'s own message for everything else', () => {
    const t = makeTranslate(zh)
    expect(failureLine(t, { code: 'refactory/console-failed', message: 'the console exited 2' }))
      .toBe('操作失败：the console exited 2')
  })
})

describe('the picker before it has read anything', () => {
  it('reads the choices on mount, which is the only read the body starts itself', async () => {
    const { console, instance } = mountBody({ bare: true })
    await waitFor(() => { expect(console.modes).toHaveBeenCalledTimes(1) })
    expect(console.strategies).toHaveBeenCalledTimes(1)
    expect(console.packages).toHaveBeenCalledTimes(1)
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({
      kind: 'ready',
      value: { project: '/repo', modes: [], strategies: NO_STRATEGIES, packages: [], projects: [] },
    })
  })

  it('says it is reading them', () => {
    const { view } = mountBody({ choices: { kind: 'loading' } })
    expect(at(view, 'data-refactory-state').getAttribute('data-refactory-state')).toBe('loading')
    expect(view.getByText(zh['choices.loading'])).toBeDefined()
  })

  it('says why the choices could not be read', () => {
    const { view } = mountBody({ choices: { kind: 'failed', message: 'the console exited 2' } })
    expect(at(view, 'data-refactory-state').getAttribute('data-refactory-state')).toBe('failed')
    expect(view.getByText('操作失败：the console exited 2')).toBeDefined()
  })
})

describe('the picker with the described choices', () => {
  it('offers packages first, with no mode chosen or run being read', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    expect(at(view, 'data-refactory-state').getAttribute('data-refactory-state')).toBe('ready')
    const select = at(view, 'data-refactory-mode') as HTMLSelectElement
    expect(select.value).toBe('')
    expect(select.disabled).toBe(true)
    expect([...select.options].map(option => option.value)).toEqual([''])
    expect([...(at(view, 'data-refactory-package') as HTMLSelectElement).options].map(option => option.value)).toEqual(['', 'default_pack', 'off', 'p'])
    expect(missing(view, 'data-refactory-start')).toBe(true)
    expect(view.getByText(zh['status.none'])).toBeDefined()
    expect(at(view, 'data-refactory-enabled', 'true').textContent).toContain('p')
    expect(at(view, 'data-refactory-package-mode', 'adaptive')).toBeDefined()
  })

  it('chooses a mode, shows what it is, and offers to start it', () => {
    const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    fireEvent.change(at(view, 'data-refactory-package'), { target: { value: 'p' } })
    fireEvent.change(at(view, 'data-refactory-mode'), { target: { value: MODE.id } })
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBe(MODE.id)
    expect(at(view, 'data-refactory-mode-note').textContent).toBe(MODE.description)
    expect(missing(view, 'data-refactory-start')).toBe(false)
    expect(missing(view, 'data-refactory-authoring')).toBe(true)
    expect(missing(view, 'data-refactory-disabled')).toBe(true)
  })

  it('choosing nothing again drops the run button', () => {
    const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    fireEvent.change(at(view, 'data-refactory-mode'), { target: { value: '' } })
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBeNull()
    expect(missing(view, 'data-refactory-start')).toBe(true)
  })

  it('says which action runs an authoring mode', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: AUTHORING.id })
    expect(at(view, 'data-refactory-mode-note').getAttribute('data-refactory-mode-note')).toBe('create')
    expect(at(view, 'data-refactory-authoring').textContent).toContain('create')
  })

  it('says when the chosen mode\'s package is disabled', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: DISABLED.id })
    expect(at(view, 'data-refactory-disabled').textContent).toBe(zh['mode.disabled'])
  })

  it('falls back to the default line for a seam whose chosen name is no longer offered', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      settings: { optimizer: 'gone' },
    })
    expect(at(view, 'data-refactory-setting-note', 'optimizer').textContent).toBe(zh['setting.default'])
  })

  it('starts the chosen mode with the settings that were chosen, and only those', async () => {
    const { view, console, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    console.start.mockImplementation(async () => ok(RECEIPT) as never)
    console.status.mockImplementation(async () => ok(FRAME) as never)
    fireEvent.change(at(view, 'data-refactory-setting', 'optimizer'), { target: { value: 'random' } })
    expect(instance.getSnapshot().byTab[TAB]?.settings.optimizer).toBe('random')
    expect(at(view, 'data-refactory-setting-note', 'optimizer').textContent).toBe('sample uniformly')
    fireEvent.click(at(view, 'data-refactory-start'))
    await waitFor(() => { expect(console.start).toHaveBeenCalledTimes(1) })
    expect(console.start).toHaveBeenCalledWith(
      { mode: MODE.id, settings: { optimizer: 'random' }, knobs: KNOBS },
      expect.anything(),
    )
    await waitFor(() => { expect(console.status).toHaveBeenCalledTimes(1) })
  })

  it('says a seam is on its default, and names the outer loops it offers', () => {
    const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    expect(at(view, 'data-refactory-setting-note', 'strategy').textContent).toBe(zh['setting.default'])
    expect(at(view, 'data-refactory-setting-note', 'evaluator').textContent).toBe(zh['setting.default'])
    const strategy = at(view, 'data-refactory-setting', 'strategy') as HTMLSelectElement
    expect([...strategy.options].map(option => option.value)).toEqual(['', 'beam'])
    fireEvent.change(strategy, { target: { value: 'beam' } })
    expect(instance.getSnapshot().byTab[TAB]?.settings.strategy).toBe('beam')
    expect(at(view, 'data-refactory-setting-note', 'strategy').textContent).toBe('keep the best few')
    fireEvent.change(at(view, 'data-refactory-setting', 'evaluator'), { target: { value: 'score' } })
    expect(instance.getSnapshot().byTab[TAB]?.settings.evaluator).toBe('score')
    expect(at(view, 'data-refactory-setting-note', 'evaluator').textContent).toBe("the mode's own score")
    const evaluator = at(view, 'data-refactory-setting', 'evaluator') as HTMLSelectElement
    expect([...evaluator.options].map(option => option.value)).toEqual(['', 'score'])
  })

  it('reads the choices again when the reader asks it to', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    fireEvent.click(at(view, 'data-refactory-reload'))
    await waitFor(() => { expect(console.modes).toHaveBeenCalledTimes(1) })
  })

  it('shows a refusal above the picker, in terms of what the reader did', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      notice: { code: 'refactory/authoring-mode', message: 'raw host text' },
    })
    const notice = at(view, 'data-refactory-notice')
    expect(notice.getAttribute('data-refactory-notice')).toBe('refactory/authoring-mode')
    expect(notice.textContent).toBe(zh['error.authoringMode'])
  })
})

describe('the run the picker is reading', () => {
  it('names the run the gesture started, so it can be followed outside the panel', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, watching: '/runs/live' })
    expect(at(view, 'data-refactory-started').textContent).toBe(zh['watching'].replace('{run}', '/runs/live'))
  })

  it('says it is reading while the first frame is on its way', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, watching: '/runs/live' })
    expect(at(view, 'data-refactory-status', 'reading').textContent).toBe(zh['status.reading'])
  })

  it('says why a status could not be read', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'failed', message: '/runs/live is not a run directory' },
    })
    expect(at(view, 'data-refactory-status', 'failed').textContent).toContain('/runs/live is not a run directory')
  })

  it('reports the run, its score, and how much it has evaluated', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
    })
    expect(at(view, 'data-refactory-run').textContent).toBe('/runs/live')
    expect(at(view, 'data-refactory-status', 'running')).toBeDefined()
    expect(view.getByText(`${zh['status.state']}: running`)).toBeDefined()
    expect(at(view, 'data-refactory-score').textContent).toBe('最好成绩 1.25')
    expect(at(view, 'data-refactory-evaluations').textContent).toBe('已评估 2 次')
    expect(missing(view, 'data-refactory-stop')).toBe(false)
  })

  it('says a run has not scored rather than showing a zero', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, best_score: null, finished: true } },
    })
    expect(at(view, 'data-refactory-score').textContent).toBe(zh['single.noAggregate'])
    expect(missing(view, 'data-refactory-stop')).toBe(true)
  })

  it('reads the next frame from where the last one reached, and asks the run to stop', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
    })
    console.status.mockImplementation(async () => ok(FRAME) as never)
    console.stop.mockImplementation(async () => ok({ stopped: true }) as never)
    fireEvent.click(at(view, 'data-refactory-refresh'))
    await waitFor(() => { expect(console.status).toHaveBeenCalledWith('/runs/live', FRAME.cursor, expect.anything()) })
    fireEvent.click(at(view, 'data-refactory-stop'))
    await waitFor(() => { expect(console.stop).toHaveBeenCalledWith('/runs/live', expect.anything()) })
  })
})

describe('the activity levels of the run the picker reads', () => {
  // T-18 (R-2.1, R-2.3-4, R-2.6-7, SD-2, UC-3): exactly three levels, standard by
  // default and visible, verbose exhausts the journal from 0, compact hides
  // detail but keeps the way to the history.
  it('offers exactly three levels with standard selected on attach', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 200, 450) } },
    })
    const levels = [...view.container.querySelectorAll('[data-refactory-activity-level]')]
    expect(levels.map(node => node.getAttribute('data-refactory-activity-level'))).toEqual(['compact', 'standard', 'verbose'])
    // A radio group is the control: one visible choice, every option focusable.
    expect([...view.container.querySelectorAll('[data-refactory-activity-level] input')]
      .map(node => (node as HTMLInputElement).checked)).toEqual([false, true, false])
    expect(at(view, 'data-refactory-activity-level', 'standard').getAttribute('data-selected')).toBe('true')
    expect(at(view, 'data-refactory-activity').getAttribute('data-refactory-level')).toBe('standard')
  })

  it('draws the newest window at standard, and a control that can reach the rest', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 200, 450) } },
    })
    // SD-2: the standard window is the newest 200 records of what is held.
    expect(rows(view)).toHaveLength(200)
    expect(missing(view, 'data-refactory-activity-earlier')).toBe(false)
    expect(missing(view, 'data-refactory-activity-more')).toBe(false)
  })

  it('exhausts the journal to has_more: false when verbose is chosen, holding index 0 through total-1', async () => {
    const { view, console, instance } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 200, 450) } },
    })
    console.status.mockImplementation(async (_run: string, after: number) => ok({ ...FRAME, activity: live(after, 200, 450) }) as never)
    fireEvent.click(at(view, 'data-refactory-activity-level', 'verbose').querySelector('input')!)
    await waitFor(() => { expect(instance.getSnapshot().byTab[TAB]?.activity?.complete).toBe(true) })
    // R-2.2: every complete record, accumulated page by page; the answer that
    // said `has_more: false` and its `total` are the completeness evidence.
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toHaveLength(450)
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events[0]).toMatchObject({ at: 't0' })
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events.at(-1)).toMatchObject({ at: 't449' })
    expect(instance.getSnapshot().byTab[TAB]?.activity?.total).toBe(450)
    expect(instance.getSnapshot().byTab[TAB]?.activityLevel).toBe('verbose')
    // The pass asked from 0 and continued from each answer's own cursor.
    expect(console.status.mock.calls.map(call => call[1])).toEqual([0, 200, 400])
  })

  it('hides the detail at compact but keeps the history and its paging control reachable', () => {
    const plain = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: { ...live(0, 200, 450), resources: { cpu: 1 }, outputs: { name: 'demo' } } } },
      activityLevel: 'compact',
    })
    // §2 R-2's table: outputs, resource usage and delivery are standard-and-above.
    expect(missing(plain.view, 'data-refactory-activity-outputs')).toBe(true)
    expect(plain.view.queryByText('Resource usage and limits')).toBeNull()
    // R-2.4 / DEC-001: the history and the way to it are drawn at every level.
    expect(missing(plain.view, 'data-refactory-activity-history')).toBe(false)
    expect(missing(plain.view, 'data-refactory-activity-earlier')).toBe(false)
    expect(rows(plain.view)).toHaveLength(200)
  })

  it('keeps the summary every level draws, at compact', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 200, 450) } },
      activityLevel: 'compact',
    })
    expect(view.getByText(/Current nodes/)).toBeDefined()
    expect(view.getByText(/Waiting/)).toBeDefined()
    expect(view.getByText(/Completed/)).toBeDefined()
  })

  // T-18b (R-2.2, R-2.5, SD-12, F-4): the DOM bound is the contract, and it is
  // independent of how many records are held.
  it('renders at most RENDER_WINDOW + 2 rows for a held history of 5,000 records', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      activityLevel: 'verbose',
      activity: { after: 0, cursor: 5000, events: ACTIVITY(0, 5000, 5000).events, total: 5000, paged: true },
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 5000, 5000) } },
    })
    expect(rows(view).length).toBeLessThanOrEqual(RENDER_WINDOW + 2)
    expect(rows(view).length).toBe(RENDER_WINDOW)
  })

  // Reported from a live 437-record run: the history opened with a large empty
  // gap above its first row. The drawn slice is translated to its position in
  // the history, but the viewport stayed scrolled to the top, so the rows began
  // far below the visible area. Rendering therefore has to position the viewport
  // at the slice, which is asserted here from the two values that must agree.
  it('positions the viewport at the slice it draws, so the rows are in view', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      activityLevel: 'standard',
      activity: { after: 0, cursor: 437, events: ACTIVITY(0, 437, 437).events, total: 437, paged: true },
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 437, 437) } },
    })
    const viewport = at(view, 'data-refactory-activity-history').querySelector('[role="group"]') as HTMLElement
    const indices = rows(view).map(row => Number(row.getAttribute('data-refactory-activity-row')))
    // Standard still opens at the newest records (SD-2)...
    expect(indices[0]).toBe(437 - RENDER_WINDOW)
    // ...and the slice index the element reports must be the one whose offset the
    // viewport is scrolled to. A view scrolled to 0 while the slice sits at row
    // 237 is the reported gap.
    const slice = Number(viewport.getAttribute('data-refactory-activity-window'))
    expect(slice).toBe(437 - RENDER_WINDOW)
    expect(viewport.getAttribute('data-refactory-activity-offset')).toBe(String(slice * 24))
  })

  it('scrolls to a row near the end, renders it, and still keeps the count bounded', () => {
    const { view, instance } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      activityLevel: 'verbose',
      activity: { after: 0, cursor: 5000, events: ACTIVITY(0, 5000, 5000).events, total: 5000, paged: true },
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 5000, 5000) } },
    })
    const viewport = at(view, 'data-refactory-activity-history').querySelector('[role="group"]') as HTMLElement
    // A window of rows down: the store keeps a row index, not a pixel offset.
    Object.defineProperty(viewport, 'scrollTop', { value: 24 * RENDER_WINDOW * 23, configurable: true })
    fireEvent.scroll(viewport)
    expect(instance.getSnapshot().byTab[TAB]?.activityScroll).toBe(23 * RENDER_WINDOW)
    expect(rows(view).length).toBeLessThanOrEqual(RENDER_WINDOW + 2)
    // The row at that position is drawn, and it is the journal index it claims.
    const indices = rows(view).map(row => Number(row.getAttribute('data-refactory-activity-row')))
    expect(indices[0]).toBe(23 * RENDER_WINDOW)
    expect(view.getByText(/t4600/)).toBeDefined()
  })

  it('re-renders the window, not the history, when a page arrives', () => {
    const { view, instance } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      activityLevel: 'verbose',
      activity: { after: 0, cursor: 400, events: ACTIVITY(0, 400, 400).events, total: 400, paged: true },
      frame: { kind: 'ready', value: { ...FRAME, activity: live(0, 400, 400) } },
    })
    const before = rows(view).length
    act(() => {
      instance.actions.activityPage(TAB, '/runs/live', {
        after: 400, cursor: 600, events: ACTIVITY(400, 200, 600).events, total: 600, paged: true, run: null,
      })
    })
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toHaveLength(600)
    expect(rows(view).length).toBe(before)
    expect(rows(view).length).toBeLessThanOrEqual(RENDER_WINDOW + 2)
  })

  // T-19 (R-1.4, F-6): "Load earlier activity" walks to index 0 and then disables.
  it('walks "Load earlier activity" to index 0 and then disables it', async () => {
    const { view, console, instance } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: live(400, 200, 600) } },
      // The tab holds records 400..599 of 600, so index 0 is two presses away.
      activity: { after: 400, cursor: 600, events: live(400, 200, 600).events, total: 600, paged: true },
    })
    console.status.mockImplementation(async (_run: string, after: number) => ok({ ...FRAME, activity: live(after, 200, 600) }) as never)
    const earlier = () => at(view, 'data-refactory-activity-earlier') as HTMLButtonElement
    expect(earlier().disabled).toBe(false)
    fireEvent.click(earlier())
    await waitFor(() => { expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toHaveLength(400) })
    expect(console.status).toHaveBeenLastCalledWith('/runs/live', 200, expect.anything())
    fireEvent.click(earlier())
    await waitFor(() => { expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toHaveLength(600) })
    expect(console.status).toHaveBeenLastCalledWith('/runs/live', 0, expect.anything())
    // Index 0 is held, so the first record is there and the control is spent.
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events[0]).toMatchObject({ at: 't0' })
    await waitFor(() => { expect(earlier().disabled).toBe(true) })
  })

  // T-20 (R-6.3, UC-9, UC-10): an older console's answer is shown whole and
  // marked unpaged rather than passed off as the complete history.
  it('draws an older console\'s whole answer with the unpaged note', () => {
    const { has_more: _more, total: _total, after: _after, page_size: _size, ...unpaged } = live(0, 3, 3)
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, activity: { ...unpaged, paged: false } } },
    })
    expect(rows(view)).toHaveLength(3)
    expect(at(view, 'data-refactory-activity-unpaged').textContent).toBe(zh['activity.unpagedNote'])
    // The answer's own page is the whole of what this console serves, so the
    // history is whole as far as it goes — but it is *marked* unpaged rather
    // than passed off as a paged history that happens to be short (R-6.3).
    expect(at(view, 'data-refactory-activity-count').textContent).toContain(zh['activity.count'].replace('{held}', '3').replace('{total}', '3'))
  })

  // T-23b (R-2 outputs row, SD-4, E-1): `outputs` stays, and so does its disclosure.
  it('keeps the "Outputs and validation" disclosure at standard and at verbose', () => {
    for (const level of ['standard', 'verbose'] as const) {
      const { view } = mountBody({
        choices: { kind: 'ready', value: CHOICES },
        watching: '/runs/live',
        activityLevel: level,
        frame: { kind: 'ready', value: { ...FRAME, activity: { ...live(0, 200, 200), outputs: { name: 'demo', validation: 'ok' } } } },
      })
      const disclosure = at(view, 'data-refactory-activity-outputs')
      expect(disclosure.querySelector('summary')?.textContent).toBe('Outputs and validation')
      expect(disclosure.querySelector('pre')?.textContent).toContain('"validation": "ok"')
      cleanup()
    }
  })

  // T-17 (R-4.2-3, F-3): refresh continues the journal, not the progress cursor.
  it('refreshes the journal from the highest held cursor rather than the progress cursor', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: { ...FRAME, cursor: 3, activity: live(200, 200, 450) } },
      activity: { after: 200, cursor: 400, events: ACTIVITY(200, 200, 450).events, total: 450, paged: true },
    })
    console.status.mockImplementation(async () => ok({ ...FRAME, activity: live(400, 200, 450) }) as never)
    fireEvent.click(at(view, 'data-refactory-refresh'))
    // 400 is the journal index the held history reaches; 3 is the progress
    // cursor, which indexes a different file and is never the journal's index.
    await waitFor(() => { expect(console.status).toHaveBeenCalledWith('/runs/live', 400, expect.anything()) })
  })

  it('renders the two failure lines differently: a cut payload is not "no JSON"', () => {
    const truncated = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'failed', code: 'refactory/console-truncated', message: 'the console output was truncated at the 4194304-byte collection cap', details: { capBytes: 4194304, retainedBytes: 4194304 } },
    })
    const cut = at(truncated.view, 'data-refactory-status', 'failed').textContent ?? ''
    expect(cut).toContain('4194304')
    expect(cut).not.toContain('no JSON')
    cleanup()
    const plain = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'failed', code: 'refactory/console-failed', message: 'the console printed no JSON: oops' },
    })
    expect(at(plain.view, 'data-refactory-status', 'failed').textContent).toContain('the console printed no JSON: oops')
  })
})

describe('the steering the picker offers for the run it reads', () => {
  it('says no run is being read when none is', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    expect(at(view, 'data-refactory-steering', 'none').textContent).toBe(zh['steering.none'])
  })

  it('says it is reading the ledger while the read is on its way', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, watching: '/runs/live' })
    expect(at(view, 'data-refactory-steering', 'reading').textContent).toBe(zh['steering.reading'])
  })

  it('draws every ledger entry, newest last, with its kind and text', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      steering: { kind: 'ready', value: STEERING },
    })
    expect(at(view, 'data-refactory-steering', 'ready')).toBeDefined()
    openLedger(view)
    const entries = [...view.container.querySelectorAll('[data-refactory-entry]')]
    expect(entries.map(node => node.getAttribute('data-refactory-entry'))).toEqual(['Initial Idea', 'correction'])
    expect(entries[0]?.textContent).toContain('make it better')
    expect(entries[1]?.textContent).toContain('use a smaller grid')
  })

  it('says the ledger has nothing in it rather than drawing an empty list', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      steering: { kind: 'ready', value: { ...STEERING, exists: false, entries: [] } },
    })
    expect(at(view, 'data-refactory-ledger', 'empty').textContent).toBe(zh['steering.empty'])
    expect(missing(view, 'data-refactory-ledger', 'entries')).toBe(true)
  })

  it('says why the ledger could not be read', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      steering: { kind: 'failed', message: 'no ledger at /runs/live' },
    })
    expect(at(view, 'data-refactory-steering', 'failed').textContent).toContain('no ledger at /runs/live')
  })

  it('sends the note the person typed, with the kind they chose, and shows what came back', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
    })
    console.steer.mockImplementation(async () => ok(STEERED) as never)
    const add = at(view, 'data-refactory-note-add') as HTMLButtonElement
    expect(add.disabled).toBe(true)
    const note = at(view, 'data-refactory-note') as HTMLInputElement
    fireEvent.change(note, { target: { value: 'try the other seed' } })
    fireEvent.change(at(view, 'data-refactory-note-kind'), { target: { value: 'correction' } })
    expect(add.disabled).toBe(false)
    fireEvent.submit(at(view, 'data-refactory-note-add').closest('form') ?? view.container)
    await waitFor(() => {
      expect(console.steer).toHaveBeenCalledWith(
        { run: '/runs/live', note: 'try the other seed', kind: 'correction' },
        expect.anything(),
      )
    })
    // The field is cleared, and the entry the Host answered with is drawn.
    await waitFor(() => { expect(note.value).toBe('') })
    openLedger(view)
    await waitFor(() => { expect(at(view, 'data-refactory-entry', 'info').textContent).toContain('try the other seed') })
  })

  it('offers the gate and the verdicts it accepts only while a gate is waiting', () => {
    const waiting = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: GATED_STEERING },
    })
    expect(at(waiting.view, 'data-refactory-gate', 'approve').textContent).toContain('the person approves the draft')
    expect([...waiting.view.container.querySelectorAll('[data-refactory-verdict]')]
      .map(node => node.getAttribute('data-refactory-verdict'))).toEqual(['proceed', 'halt'])
    const quiet = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
    })
    expect(missing(quiet.view, 'data-refactory-gate')).toBe(true)
    expect(missing(quiet.view, 'data-refactory-verdict')).toBe(true)
  })

  it('keeps the line breaks of a gate\'s prompt', () => {
    // Design mode's card is one item a line; a plain note collapses them (dm-163).
    const prompt = 'Design: it prints.\n\nPitfalls: 1\n- PIT-001 (high): addressed\n\nApprove the design?'
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: { ...GATED_STEERING, gates: [{ ...GATED_STEERING.gates[0]!, prompt }] } },
    })
    const note = at(view, 'data-refactory-gate-note')
    expect(note.textContent).toBe(prompt)
    expect(note.classList.contains(css.gatePrompt!)).toBe(true)
    // jsdom applies no module CSS, so the rule is read where it is written (vitest runs
    // from the repository root).
    const rules = readFileSync('packages/client/ui-refactory/src/client/RefactoryBody.module.css', 'utf8')
    expect(rules).toMatch(/\.gatePrompt\s*\{[^}]*white-space:\s*pre-wrap/u)
  })

  it('answers the waiting gate with the verdict the person pressed', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: GATED_STEERING },
    })
    console.steer.mockImplementation(async () => ok(GATED_STEERING) as never)
    fireEvent.click(at(view, 'data-refactory-verdict', 'proceed'))
    await waitFor(() => {
      expect(console.steer).toHaveBeenCalledWith(
        { run: '/runs/live', verdicts: ['approve=proceed'] },
        expect.anything(),
      )
    })
  })

  it('offers pause, resume, stop and adopt, and sends the one pressed', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
    })
    console.steer.mockImplementation(async () => ok(STEERED) as never)
    expect([...view.container.querySelectorAll('[data-refactory-control]')]
      .map(node => node.getAttribute('data-refactory-control')))
      .toEqual(['pause', 'resume', 'stop', 'adopt'])
    fireEvent.click(at(view, 'data-refactory-control', 'pause'))
    await waitFor(() => {
      expect(console.steer).toHaveBeenCalledWith(
        { run: '/runs/live', control: 'pause' },
        expect.anything(),
      )
    })
  })

  it('offers the candidates with their flags, and sends the act a row asks for', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
      candidates: { kind: 'ready', value: CANDIDATES },
    })
    console.steer.mockImplementation(async () => ok(STEERED) as never)
    console.leaderboard.mockImplementation(async () => ok(CANDIDATES) as never)
    const rows = [...view.container.querySelectorAll('[data-refactory-candidate]')]
    expect(rows.map(row => row.getAttribute('data-refactory-candidate')))
      .toEqual(['aaaa11', 'bbbb22', 'cccc33'])
    expect(rows.map(row => row.getAttribute('data-refactory-kept')))
      .toEqual(['false', 'true', 'false'])
    expect(rows.map(row => row.getAttribute('data-refactory-rejected')))
      .toEqual(['false', 'false', 'true'])
    // A candidate already kept cannot be kept again, and one already rejected
    // cannot be rejected again; either act is one the console would refuse as a
    // second request in a file that holds one.
    expect((at(view, 'data-refactory-candidate-keep', 'bbbb22') as HTMLButtonElement).disabled).toBe(true)
    expect((at(view, 'data-refactory-candidate-reject', 'cccc33') as HTMLButtonElement).disabled).toBe(true)

    fireEvent.click(at(view, 'data-refactory-candidate-reject', 'aaaa11'))
    await waitFor(() => {
      expect(console.steer).toHaveBeenCalledWith(
        { run: '/runs/live', control: 'reject', target: 'aaaa11' },
        expect.anything(),
      )
    })
  })

  it('adopts the best candidate now, and reads the table it acted on again', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
      candidates: { kind: 'ready', value: CANDIDATES },
    })
    console.steer.mockImplementation(async () => ok(STEERED) as never)
    console.leaderboard.mockImplementation(async () => ok(CANDIDATES) as never)
    fireEvent.click(at(view, 'data-refactory-control', 'adopt'))
    await waitFor(() => {
      expect(console.steer).toHaveBeenCalledWith(
        { run: '/runs/live', control: 'adopt' },
        expect.anything(),
      )
      expect(console.leaderboard).toHaveBeenCalledWith('/runs/live', expect.anything())
    })
  })

  it('reads the candidates on demand, and says when there are none', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME },
      steering: { kind: 'ready', value: STEERING },
    })
    console.leaderboard.mockImplementation(async () => ok([]) as never)
    // Absent until asked for: a leaderboard is a console call of its own, and a
    // search nobody is steering does not need its table fetched every tick.
    expect(missing(view, 'data-refactory-candidates-rows')).toBe(true)
    fireEvent.click(at(view, 'data-refactory-candidates-read'))
    await waitFor(() => { expect(console.leaderboard).toHaveBeenCalledWith('/runs/live', expect.anything()) })
    await waitFor(() => {
      expect(at(view, 'data-refactory-candidates-empty').textContent).toBe(zh['candidates.empty'])
    })
  })

  it('reads the ledger again when the reader asks it to', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      watching: '/runs/live',
      steering: { kind: 'ready', value: STEERING },
    })
    console.steer.mockImplementation(async () => ok(STEERING) as never)
    fireEvent.click(at(view, 'data-refactory-steering-refresh'))
    await waitFor(() => { expect(console.steer).toHaveBeenCalledWith({ run: '/runs/live' }, expect.anything()) })
  })
})

describe('the knobs the chosen mode declares', () => {
  it('draws one field per declared knob, at the value the declaration gives it', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    expect(view.container.querySelectorAll('[data-refactory-knob-input]')).toHaveLength(5)
    expect([...view.container.querySelectorAll('[data-refactory-knob-input]')]
      .map(node => node.getAttribute('data-refactory-knob-input')))
      .toEqual(['rounds', 'temperature', 'source_root', 'preset', 'verbose'])
    expect((at(view, 'data-refactory-knob-input', 'rounds') as HTMLInputElement).value).toBe('3')
    expect((at(view, 'data-refactory-knob-input', 'source_root') as HTMLInputElement).value).toBe('/tmp/src')
  })

  it('draws a declared set as a choice, a bool as the console\'s two words, and a number as a number', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    const preset = at(view, 'data-refactory-knob-input', 'preset') as HTMLSelectElement
    expect([...preset.options].map(option => option.value)).toEqual(['broad', 'narrow'])
    const verbose = at(view, 'data-refactory-knob-input', 'verbose') as HTMLSelectElement
    expect([...verbose.options].map(option => option.value)).toEqual(['true', 'false'])
    const rounds = at(view, 'data-refactory-knob-input', 'rounds') as HTMLInputElement
    expect([rounds.type, rounds.step, rounds.min, rounds.max]).toEqual(['number', '1', '1', '10'])
    // An unbounded float takes any value, so it states no bounds at all.
    const temperature = at(view, 'data-refactory-knob-input', 'temperature') as HTMLInputElement
    expect([temperature.type, temperature.step, temperature.hasAttribute('min'), temperature.hasAttribute('max')])
      .toEqual(['number', 'any', false, false])
    expect((at(view, 'data-refactory-knob-input', 'source_root') as HTMLInputElement).type).toBe('text')
  })

  it('says what a knob is for, and marks one the search may not move', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    expect(at(view, 'data-refactory-knob-description', 'rounds').textContent).toBe('how many rounds')
    expect(missing(view, 'data-refactory-knob-description', 'temperature')).toBe(true)
    expect(at(view, 'data-refactory-knob-frozen', 'source_root').textContent).toBe(zh['knob.frozen'])
    expect(missing(view, 'data-refactory-knob-frozen', 'rounds')).toBe(true)
  })

  it('changes a knob value through the store and sends it with the run', async () => {
    const { view, console, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    console.start.mockImplementation(async () => ok(RECEIPT) as never)
    console.status.mockImplementation(async () => ok(FRAME) as never)
    fireEvent.change(at(view, 'data-refactory-knob-input', 'rounds'), { target: { value: '7' } })
    expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({ rounds: '7' })
    fireEvent.click(at(view, 'data-refactory-start'))
    await waitFor(() => { expect(console.start).toHaveBeenCalledTimes(1) })
    expect(console.start).toHaveBeenCalledWith({
      mode: MODE.id,
      settings: {},
      knobs: ['rounds=7', 'temperature=0.5', 'source_root=/tmp/src', 'preset=broad', 'verbose=false'],
    }, expect.anything())
  })

  it('takes a chosen value and typed text through the same store', () => {
    const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    fireEvent.change(at(view, 'data-refactory-knob-input', 'preset'), { target: { value: 'narrow' } })
    fireEvent.change(at(view, 'data-refactory-knob-input', 'source_root'), { target: { value: '/tmp/other' } })
    fireEvent.change(at(view, 'data-refactory-knob-input', 'verbose'), { target: { value: 'true' } })
    expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({
      preset: 'narrow',
      source_root: '/tmp/other',
      verbose: 'true',
    })
    expect((at(view, 'data-refactory-knob-input', 'preset') as HTMLSelectElement).value).toBe('narrow')
    expect((at(view, 'data-refactory-knob-input', 'source_root') as HTMLInputElement).value).toBe('/tmp/other')
  })
})

describe('using the chosen mode once', () => {
  it('offers Run once beside Start for a mode the console says start runs', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    expect(missing(view, 'data-refactory-run-once')).toBe(false)
    expect(missing(view, 'data-refactory-start')).toBe(false)
  })

  it('offers authoring Run once and keeps optimizer Start disabled', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: AUTHORING.id })
    console.run.mockImplementation(async () => ok({ ...SINGLE, mode: AUTHORING.id, best_score: null }) as never)
    expect((at(view, 'data-refactory-start') as HTMLButtonElement).disabled).toBe(true)
    expect((at(view, 'data-refactory-run-once') as HTMLButtonElement).disabled).toBe(false)
    expect(at(view, 'data-refactory-authoring').textContent).toContain('create')
    fireEvent.click(at(view, 'data-refactory-run-once'))
    await waitFor(() => { expect(console.run).toHaveBeenCalledWith(expect.objectContaining({ mode: AUTHORING.id }), expect.anything()) })
  })

  it('uses the mode once with the same knob values Start would send', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    console.run.mockImplementation(async () => ok(SINGLE) as never)
    fireEvent.change(at(view, 'data-refactory-knob-input', 'rounds'), { target: { value: '7' } })
    fireEvent.click(at(view, 'data-refactory-run-once'))
    await waitFor(() => { expect(console.run).toHaveBeenCalledTimes(1) })
    expect(console.run).toHaveBeenCalledWith({
      mode: MODE.id,
      knobs: ['rounds=7', 'temperature=0.5', 'source_root=/tmp/src', 'preset=broad', 'verbose=false'],
    }, expect.anything())
  })

  it('says it is going while the run is in flight, and keeps the button unpressable', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'running' },
    })
    expect(at(view, 'data-refactory-single', 'running').textContent).toBe(zh['single.running'])
    expect((at(view, 'data-refactory-run-once') as HTMLButtonElement).disabled).toBe(true)
  })

  it('reports the run, its score and what it recorded', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'ready', value: SINGLE },
    })
    expect(at(view, 'data-refactory-single', SINGLE.state)).toBeDefined()
    expect(at(view, 'data-refactory-single-run').textContent).toBe(zh['single.ran'].replace('{mode}', SINGLE.mode ?? SINGLE.run_id).replace('{run}', SINGLE.root))
    expect(view.getByText(`${zh['status.state']}: finished`)).toBeDefined()
    expect(at(view, 'data-refactory-single-score').textContent).toBe('最好成绩 0.75')
    expect(at(view, 'data-refactory-fact', 'name').textContent).toBe('name: demo')
  })

  it('says a run stopped at a gate a person owns is waiting on a person, not failing', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'ready', value: GATED },
    })
    expect(at(view, 'data-refactory-single-spine', 'ready').textContent)
      .toBe(`${zh['single.spine']}: ready`)
    expect(at(view, 'data-refactory-single-due').textContent)
      .toBe(zh['single.due'].replace('{nodes}', 'approve, overwatch'))
    expect(at(view, 'data-refactory-single-gate').textContent).toBe(zh['single.gate'])
    expect(at(view, 'data-refactory-single-score').textContent).toBe(zh['status.noScore'])
  })

  it('reports a gate the spine refused with the spine\'s own reason', () => {
    const refused = {
      ...SINGLE,
      state: 'failed',
      finished: false,
      authoring: { spine: 'refused', nodes_due: [], error: 'gate approve is answered by a person' },
    }
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'ready', value: refused },
    })
    expect(at(view, 'data-refactory-single-spine', 'refused')).toBeDefined()
    expect(at(view, 'data-refactory-single-gate')).toBeDefined()
    expect(at(view, 'data-refactory-single-error').textContent).toContain('gate approve is answered by a person')
    expect(missing(view, 'data-refactory-single-due')).toBe(true)
  })

  it('draws no gate line or fact list for a run that simply finished', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'ready', value: COMPLETED },
    })
    expect(at(view, 'data-refactory-single-spine', 'done')).toBeDefined()
    expect(missing(view, 'data-refactory-single-gate')).toBe(true)
    expect(missing(view, 'data-refactory-fact')).toBe(true)
  })

  it('draws no graph status for a run the spine did not drive', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      chosen: MODE.id,
      single: { kind: 'ready', value: { ...SINGLE, authoring: null } },
    })
    expect(missing(view, 'data-refactory-single-spine')).toBe(true)
    expect(missing(view, 'data-refactory-single-gate')).toBe(true)
  })

  it('shows a refused gesture as the notice, not as a run result', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    console.run.mockImplementation(async () => ({
      ok: false,
      error: { code: 'refactory/authoring-mode', message: 'raw host text' },
    }) as never)
    fireEvent.click(at(view, 'data-refactory-run-once'))
    await waitFor(() => { expect(at(view, 'data-refactory-notice').textContent).toContain('raw host text') })
    expect(missing(view, 'data-refactory-single')).toBe(true)
    expect((at(view, 'data-refactory-run-once') as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('the packages the picker manages', () => {
  it('offers to disable an enabled package and to enable a disabled one', async () => {
    const { view, console } = mountBody({
      choices: { kind: 'ready', value: { ...CHOICES, packages: [PACKAGE, { ...PACKAGE, name: 'off', enabled: false }] } },
    })
    console.setPackageEnabled.mockImplementation(async () => ok([]) as never)
    expect(at(view, 'data-refactory-toggle', 'p').textContent).toBe(zh['packages.disable'])
    expect(at(view, 'data-refactory-toggle', 'off').textContent).toBe(zh['packages.enable'])
    fireEvent.click(at(view, 'data-refactory-toggle', 'off'))
    await waitFor(() => { expect(console.setPackageEnabled).toHaveBeenCalledWith('off', true, expect.anything()) })
  })

  it('offers no toggle for a package that reported no state', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: { ...CHOICES, packages: [{ ...PACKAGE, enabled: null, modes: [] }] } },
    })
    expect(missing(view, 'data-refactory-toggle')).toBe(true)
    expect(at(view, 'data-refactory-enabled', 'unknown').textContent).toContain(zh['packages.unknown'])
    expect(view.getByText('0 个模式')).toBeDefined()
  })

  it('waits out a package change, keeping its buttons unpressable', () => {
    const { view } = mountBody({
      choices: { kind: 'ready', value: CHOICES },
      busy: true,
    })
    expect((at(view, 'data-refactory-remove', 'p') as HTMLButtonElement).disabled).toBe(true)
    expect((at(view, 'data-refactory-import') as HTMLButtonElement).disabled).toBe(true)
  })

  it('removes a package', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    console.removePackage.mockImplementation(async () => ok([]) as never)
    fireEvent.click(at(view, 'data-refactory-remove', 'p'))
    await waitFor(() => { expect(console.removePackage).toHaveBeenCalledWith('p', expect.anything()) })
  })

  it('imports from a directory or git URL, clearing the fields it sent', async () => {
    const { view, console } = mountBody({ choices: { kind: 'ready', value: CHOICES } })
    console.importPackage.mockImplementation(async () => ok([PACKAGE]) as never)
    const source = at(view, 'data-refactory-import-source') as HTMLInputElement
    const name = at(view, 'data-refactory-import-name') as HTMLInputElement
    expect((at(view, 'data-refactory-import') as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(source, { target: { value: '/src/pack' } })
    fireEvent.change(name, { target: { value: 'renamed' } })
    expect((at(view, 'data-refactory-import') as HTMLButtonElement).disabled).toBe(false)
    fireEvent.submit(at(view, 'data-refactory-import').closest('form') ?? view.container)
    await waitFor(() => { expect(console.importPackage).toHaveBeenCalledWith('/src/pack', 'renamed', expect.anything()) })
    expect(source.value).toBe('')
    expect(name.value).toBe('')
  })

  it('draws every string from the dictionary the renderer bound', () => {
    const english = mountBody({ choices: { kind: 'ready', value: CHOICES }, dictionary: en })
    expect(english.view.getByText(en['run.heading'])).toBeDefined()
    expect(english.view.getByText(en['packages.heading'])).toBeDefined()
    // Three seams, so the default is named once per select and once per note.
    expect(english.view.getAllByText(en['setting.default'])).toHaveLength(6)
    expect(english.view.queryByText(zh['setting.default'])).toBeNull()
  })
})


describe('panel disclosure and launch affordances', () => {
  it('retains the same input and edited value when mode inputs are folded', () => {
    const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id })
    const input = at(view, 'data-refactory-knob-input', 'rounds') as HTMLInputElement
    fireEvent.change(input, { target: { value: '7' } })
    const details = input.closest('details')!
    expect(details.open).toBe(true)
    fireEvent.click(details.querySelector('summary')!)
    expect(details.open).toBe(false)
    fireEvent.click(details.querySelector('summary')!)
    expect(at(view, 'data-refactory-knob-input', 'rounds')).toBe(input)
    expect(input.value).toBe('7')
    expect(instance.getSnapshot().byTab[TAB]?.knobs.rounds).toBe('7')
  })

  it('puts a direct question link above the configuration without hiding verdicts', () => {
    const { view } = mountBody({ choices: { kind: 'ready', value: CHOICES }, watching: '/runs/live',
      frame: { kind: 'ready', value: FRAME }, steering: { kind: 'ready', value: GATED_STEERING } })
    const link = view.getByRole('link', { name: zh['panel.answer'] })
    const target = view.container.querySelector(link.getAttribute('href')!)!
    expect(target.querySelector('[data-refactory-verdict="proceed"]')).not.toBeNull()
    expect(link.compareDocumentPosition(at(view, 'data-refactory-mode')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})


it('filters modes by package and keeps the watched run when the selection changes', () => {
  const { view, instance } = mountBody({ choices: { kind: 'ready', value: CHOICES }, chosen: MODE.id,
    knobs: { rounds: '7' }, watching: '/runs/live', frame: { kind: 'ready', value: FRAME } })
  fireEvent.change(at(view, 'data-refactory-package'), { target: { value: 'default_pack' } })
  expect([...(at(view, 'data-refactory-mode') as HTMLSelectElement).options].map(option => option.value)).toEqual(['', AUTHORING.id])
  expect(instance.getSnapshot().byTab[TAB]?.chosen).toBeNull()
  expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({})
  expect(instance.getSnapshot().byTab[TAB]?.watching).toBe('/runs/live')
  fireEvent.change(at(view, 'data-refactory-mode'), { target: { value: AUTHORING.id } })
  expect(instance.getSnapshot().byTab[TAB]?.chosen).toBe(AUTHORING.id)
  expect((at(view, 'data-refactory-start') as HTMLButtonElement).disabled).toBe(true)
})
