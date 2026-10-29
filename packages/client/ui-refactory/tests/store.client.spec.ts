/**
 * The picker's store: every action's write, and the bucket rule that ties a
 * write to the tab that is still on screen.
 */
import { describe, expect, it } from 'vitest'
import type { TabId } from '@deepseek-ai/dsh-client-ui-dockkit'
import { NO_SETTINGS, createRefactoryStore, knobValues } from '../src/client/store.ts'
import { CANDIDATES, FRAME, GATED_STEERING, MODE, NO_STRATEGIES, PACKAGE, PROJECT, SINGLE, STEERING } from './fake-console.client.ts'

const TAB = 'tab-1' as TabId
const OTHER = 'tab-2' as TabId

/** Three journal records, as an answer carries them. */
const EVENTS: readonly Readonly<Record<string, unknown>>[] = [
  { at: '2026-01-01T00:00:00', kind: 'worker_event', node_id: 'a' },
  { at: '2026-01-01T00:00:01', kind: 'worker_event', node_id: 'b' },
  { at: '2026-01-01T00:00:02', kind: 'worker_event', node_id: 'c' },
]

const CHOICES = { project: '/repo', modes: [MODE], strategies: { ...NO_STRATEGIES, optimizers: [{ name: 'random', description: 'sample uniformly' }] }, packages: [PACKAGE], projects: [PROJECT] }

function open() {
  const instance = createRefactoryStore().create()
  instance.actions.start(TAB)
  return instance
}

describe('createRefactoryStore', () => {
  it('seeds a tab with the choices being read, nothing held, and the standard level', () => {
    const instance = open()
    expect(instance.getSnapshot().byTab[TAB]).toEqual({
      choices: { kind: 'loading' },
      chosen: null,
      selectedPackage: null,
      setupIntent: 'run',
      knobs: {},
      settings: { ...NO_SETTINGS },
      watching: null,
      frame: null,
      activity: null,
      activityLevel: 'standard',
      activityScroll: 0,
      steering: null,
      candidates: null,
      single: null,
      notice: null,
      busy: false,
      starting: false,
    })
  })

  it('records the choices the Host answered with', () => {
    const instance = open()
    instance.actions.loaded(TAB, CHOICES)
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({ kind: 'ready', value: CHOICES })
  })

  it('marks a reload as in flight, keeping what the tab has chosen', () => {
    const instance = open()
    instance.actions.loaded(TAB, CHOICES)
    instance.actions.chose(TAB, MODE.id)
    instance.actions.loading(TAB)
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({ kind: 'loading' })
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBe(MODE.id)
  })

  it('records why the choices could not be read, and drops a selection they may not contain', () => {
    const instance = open()
    instance.actions.loaded(TAB, CHOICES)
    instance.actions.chose(TAB, MODE.id)
    instance.actions.knob(TAB, 'rounds', '7')
    instance.actions.ran(TAB, SINGLE)
    instance.actions.failed(TAB, 'the console is missing')
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({ kind: 'failed', message: 'the console is missing' })
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBeNull()
    expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({})
    expect(instance.getSnapshot().byTab[TAB]?.single).toEqual({ kind: 'ready', value: SINGLE })
  })

  it('chooses one of the described modes, and can choose nothing again', () => {
    const instance = open()
    instance.actions.chose(TAB, MODE.id)
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBe(MODE.id)
    instance.actions.chose(TAB, null)
    expect(instance.getSnapshot().byTab[TAB]?.chosen).toBeNull()
  })

  it("drops the last mode's knobs but preserves the separately owned one-shot result", () => {
    const instance = open()
    instance.actions.loaded(TAB, CHOICES)
    instance.actions.chose(TAB, MODE.id)
    instance.actions.knob(TAB, 'rounds', '7')
    instance.actions.ran(TAB, SINGLE)
    instance.actions.chose(TAB, null)
    expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({})
    expect(instance.getSnapshot().byTab[TAB]?.single).toEqual({ kind: 'ready', value: SINGLE })
  })

  it('records each knob value a person changed, by name', () => {
    const instance = open()
    instance.actions.knob(TAB, 'rounds', '7')
    instance.actions.knob(TAB, 'source_root', '/tmp/pkg')
    expect(instance.getSnapshot().byTab[TAB]?.knobs).toEqual({ rounds: '7', source_root: '/tmp/pkg' })
  })

  it('chooses one level-two name per seam', () => {
    const instance = open()
    instance.actions.setting(TAB, 'strategy', 'beam')
    instance.actions.setting(TAB, 'optimizer', 'random')
    instance.actions.setting(TAB, 'evaluator', '')
    expect(instance.getSnapshot().byTab[TAB]?.settings).toEqual({ strategy: 'beam', optimizer: 'random', evaluator: '' })
  })

  it('begins reading a run, clearing the last refusal and asking for its first frame', () => {
    const instance = open()
    instance.actions.noticed(TAB, { code: 'refactory/unknown-mode', message: 'no' })
    instance.actions.watching(TAB, '/runs/live')
    expect(instance.getSnapshot().byTab[TAB]).toMatchObject({
      watching: '/runs/live',
      frame: { kind: 'loading' },
      notice: null,
    })
  })

  it('records one status frame, and the reading of the next', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.framed(TAB, FRAME)
    expect(instance.getSnapshot().byTab[TAB]?.frame).toEqual({ kind: 'ready', value: FRAME })
    instance.actions.frameLoading(TAB)
    expect(instance.getSnapshot().byTab[TAB]?.frame).toEqual({ kind: 'loading' })
  })

  it('records why the watched run\'s status could not be read', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.frameFailed(TAB, '/runs/live is not a run directory')
    expect(instance.getSnapshot().byTab[TAB]?.frame).toEqual({
      kind: 'failed',
      message: '/runs/live is not a run directory',
    })
  })

  it('reads the watched run\'s ledger, and records why it could not be read', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    expect(instance.getSnapshot().byTab[TAB]?.steering).toEqual({ kind: 'loading' })
    instance.actions.steered(TAB, STEERING)
    expect(instance.getSnapshot().byTab[TAB]?.steering).toEqual({ kind: 'ready', value: STEERING })
    instance.actions.steered(TAB, GATED_STEERING)
    expect(instance.getSnapshot().byTab[TAB]?.steering).toEqual({ kind: 'ready', value: GATED_STEERING })
    instance.actions.steeringFailed(TAB, '/runs/live is not a run directory')
    expect(instance.getSnapshot().byTab[TAB]?.steering).toEqual({
      kind: 'failed',
      message: '/runs/live is not a run directory',
    })
  })

  it('starts the watched run\'s candidates absent, then reads them', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    // Absent rather than loading: a candidate table is read on demand, so nothing
    // is in flight until a person asks for it.
    expect(instance.getSnapshot().byTab[TAB]?.candidates).toBeNull()
    instance.actions.candidatesLoading(TAB)
    expect(instance.getSnapshot().byTab[TAB]?.candidates).toEqual({ kind: 'loading' })
    instance.actions.candidatesRead(TAB, CANDIDATES)
    expect(instance.getSnapshot().byTab[TAB]?.candidates).toEqual({ kind: 'ready', value: CANDIDATES })
    instance.actions.candidatesFailed(TAB, '/runs/live has no candidates')
    expect(instance.getSnapshot().byTab[TAB]?.candidates).toEqual({
      kind: 'failed',
      message: '/runs/live has no candidates',
    })
  })

  it('records a one-shot run being under way, what it left, and that it was refused', () => {
    const instance = open()
    instance.actions.runLoading(TAB)
    expect(instance.getSnapshot().byTab[TAB]?.single).toEqual({ kind: 'running' })
    instance.actions.ran(TAB, SINGLE)
    expect(instance.getSnapshot().byTab[TAB]?.single).toEqual({ kind: 'ready', value: SINGLE })
    instance.actions.runFailed(TAB)
    expect(instance.getSnapshot().byTab[TAB]?.single).toBeNull()
  })

  it('replaces the packages a package gesture answered with, keeping the choices already read', () => {
    const instance = open()
    instance.actions.loaded(TAB, CHOICES)
    instance.actions.packages(TAB, [])
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({ kind: 'ready', value: { ...CHOICES, packages: [] } })
  })

  it('drops a package list that arrives before the first read lands', () => {
    const instance = open()
    instance.actions.packages(TAB, [PACKAGE])
    expect(instance.getSnapshot().byTab[TAB]?.choices).toEqual({ kind: 'loading' })
  })

  it('records what the last refused gesture said, and clears it', () => {
    const instance = open()
    instance.actions.noticed(TAB, { code: 'refactory/console-failed', message: 'the console exited 2' })
    expect(instance.getSnapshot().byTab[TAB]?.notice).toEqual({ code: 'refactory/console-failed', message: 'the console exited 2' })
    instance.actions.noticed(TAB, null)
    expect(instance.getSnapshot().byTab[TAB]?.notice).toBeNull()
  })

  it('records whether a package change is in flight', () => {
    const instance = open()
    instance.actions.busy(TAB, true)
    expect(instance.getSnapshot().byTab[TAB]?.busy).toBe(true)
    instance.actions.busy(TAB, false)
    expect(instance.getSnapshot().byTab[TAB]?.busy).toBe(false)
  })

  it('forgets one tab and keeps the others', () => {
    const instance = open()
    instance.actions.start(OTHER)
    instance.actions.forget(TAB)
    expect(instance.getSnapshot().byTab[TAB]).toBeUndefined()
    expect(instance.getSnapshot().byTab[OTHER]).toBeDefined()
  })

  // T-21 (R-2.1, §4.5): the level is one tab's, starts at `standard`, and a
  // forgotten tab takes the history it held with it.
  it('keeps the activity level per tab, defaulting to standard', () => {
    const instance = open()
    instance.actions.start(OTHER)
    expect(instance.getSnapshot().byTab[TAB]?.activityLevel).toBe('standard')
    instance.actions.setActivityLevel(TAB, 'verbose')
    expect(instance.getSnapshot().byTab[TAB]?.activityLevel).toBe('verbose')
    expect(instance.getSnapshot().byTab[OTHER]?.activityLevel).toBe('standard')
    instance.actions.setActivityLevel(TAB, 'compact')
    expect(instance.getSnapshot().byTab[TAB]?.activityLevel).toBe('compact')
  })

  it('keeps the level across a run switch but drops the history it held', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.activityPage(TAB, '/runs/live', { after: 0, cursor: 2, events: EVENTS.slice(0, 2), total: 2, paged: true, run: null })
    instance.actions.setActivityLevel(TAB, 'verbose')
    instance.actions.activityScrolled(TAB, 40)
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toHaveLength(2)
    instance.actions.watching(TAB, '/runs/other')
    // R-4.4: another run's indices are not this history's, so the history goes.
    expect(instance.getSnapshot().byTab[TAB]?.activity).toBeNull()
    expect(instance.getSnapshot().byTab[TAB]?.activityScroll).toBe(0)
    // R-2.1: which detail a person wants is about them, not about the run.
    expect(instance.getSnapshot().byTab[TAB]?.activityLevel).toBe('verbose')
  })

  it('forgets a tab and the history it held', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.activityPage(TAB, '/runs/live', { after: 0, cursor: 2, events: EVENTS.slice(0, 2), total: 2, paged: true, run: null })
    instance.actions.forget(TAB)
    expect(instance.getSnapshot().byTab[TAB]).toBeUndefined()
  })

  it('splices a forward page where its own after says, and marks completeness only on has_more: false', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.activityPage(TAB, '/runs/live', { after: 0, cursor: 2, events: EVENTS.slice(0, 2), total: 3, paged: true, run: null })
    expect(instance.getSnapshot().byTab[TAB]?.activity).toMatchObject({ through: 2, total: 3, complete: false, paged: true })
    instance.actions.activityPage(TAB, '/runs/live', { after: 2, cursor: 3, events: EVENTS.slice(2), total: 3, paged: true, run: null })
    expect(instance.getSnapshot().byTab[TAB]?.activity).toMatchObject({
      events: EVENTS, through: 3, total: 3, complete: true,
    })
  })

  it('prepends an earlier page and keeps the reader on the row they were reading', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.activityPage(TAB, '/runs/live', { after: 2, cursor: 3, events: EVENTS.slice(2), total: 3, paged: true, run: null })
    instance.actions.activityScrolled(TAB, 1)
    instance.actions.activityEarlier(TAB, '/runs/live', { after: 0, cursor: 2, events: EVENTS.slice(0, 2), total: 3, paged: true, run: null })
    expect(instance.getSnapshot().byTab[TAB]?.activity?.events).toEqual(EVENTS)
    // Two records were added in front of the row, so the row index moved by two.
    expect(instance.getSnapshot().byTab[TAB]?.activityScroll).toBe(3)
  })

  it('records a page in flight and why it failed, keeping what is already held', () => {
    const instance = open()
    instance.actions.watching(TAB, '/runs/live')
    instance.actions.activityPage(TAB, '/runs/live', { after: 0, cursor: 2, events: EVENTS.slice(0, 2), total: 3, paged: true, run: null })
    instance.actions.activityPageLoading(TAB, '/runs/live')
    expect(instance.getSnapshot().byTab[TAB]?.activity).toMatchObject({ reading: true, error: null })
    instance.actions.activityPageFailed(TAB, '/runs/live', 'the console exited 2')
    expect(instance.getSnapshot().byTab[TAB]?.activity).toMatchObject({
      events: EVENTS.slice(0, 2), reading: false, error: 'the console exited 2',
    })
  })

  it('refuses to write a tab it does not hold', () => {
    const instance = open()
    expect(() => { instance.actions.framed(OTHER, FRAME) }).toThrow('ui-refactory: no picker for tab "tab-2"')
  })
})

describe('knobValues', () => {
  /** The values as one map, which is the order and the content a gesture sends. */
  const sent = (values: readonly { name: string; value: string }[]) =>
    Object.fromEntries(values.map(entry => [entry.name, entry.value]))

  it('takes each declaration\'s own value under anything a person changed', () => {
    expect(sent(knobValues(MODE, { rounds: '7' }))).toEqual({
      rounds: '7',
      temperature: '0.5',
      source_root: '/tmp/src',
      preset: 'broad',
      verbose: 'false',
    })
  })

  it('keeps an emptied value, which is a value a str knob accepts', () => {
    expect(sent(knobValues(MODE, { source_root: '' })).source_root).toBe('')
  })

  it('takes every declaration\'s own value before anything is typed', () => {
    expect(sent(knobValues(MODE, undefined)).rounds).toBe('3')
  })

  it('has nothing to send before a mode is chosen', () => {
    expect(knobValues(undefined, { rounds: '7' })).toEqual([])
  })
})
