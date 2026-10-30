/**
 * The picker's face: what one gesture reads, what it writes, and what a retired
 * read is not allowed to write.
 *
 * The Remote face is scripted and the store is real, so a spec states what the
 * Host answered with and reads the store's own state — the same handoff the body
 * sees.
 */
import { describe, expect, it, vi } from 'vitest'
import type { Mock } from 'vitest'
import type { RemoteResult } from '@deepseek-ai/dsh-api-remotes/client'
import type {
  RefactoryModeChoice,
  RefactoryPackageChoice,
} from '@deepseek-ai/dsh-refactory-console/types'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { TabId } from '@deepseek-ai/dsh-client-ui-dockkit'
import { refactoryFace } from '../src/client/face.ts'
import { NO_SETTINGS, createRefactoryStore } from '../src/client/store.ts'
import type { RefactoryChoices } from '../src/client/store.ts'
import {
  ACTIVITY, AUTHORING, CANDIDATES, FRAME, GATED, GATED_STEERING, MODE, PACKAGE, RECEIPT, SINGLE,
  STEERED, STEERING, STRATEGIES,
  failure, fakeConsole, ok, settle,
} from './fake-console.client.ts'

const SESSION = 's-test' as SessionId
const TAB = 'tab-1' as TabId

const CHOICES: RefactoryChoices = { project: '/repo', modes: [MODE], strategies: STRATEGIES, packages: [PACKAGE], projects: [] }

type StoreInstance = ReturnType<ReturnType<typeof createRefactoryStore>['create']>

/** One promise a spec settles by hand, so a read can be retired before it lands. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settlePromise) => { resolve = settlePromise })
  return { promise, resolve }
}

/** Take the answer one scripted method resolves with for the rest of the spec. */
function answer(mock: Mock, result: RemoteResult<unknown>): void {
  mock.mockImplementation(async () => result as never)
}

function harness() {
  const instance = createRefactoryStore().create()
  const scripted = fakeConsole()
  const face = refactoryFace(scripted.remote)(SESSION, instance.actions)
  const controller = new AbortController()
  instance.actions.start(TAB)
  return { instance, scripted, face, signal: controller.signal, controller }
}

function tab(instance: StoreInstance) {
  return instance.getSnapshot().byTab[TAB]
}

describe('refactoryFace — reading the described choices', () => {
  it('seeds the tab and writes the three catalogues the Host answered with', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.modes, ok([MODE]))
    answer(scripted.strategies, ok(STRATEGIES))
    answer(scripted.packages, ok([PACKAGE]))
    face.load(TAB, signal)
    expect(tab(instance)?.choices).toEqual({ kind: 'loading' })
    await settle()
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: CHOICES })
    expect(scripted.modes).toHaveBeenCalledWith(signal)
    expect(scripted.strategies).toHaveBeenCalledWith(signal)
    expect(scripted.packages).toHaveBeenCalledWith(signal)
  })

  it('records the reason the mode catalogue could not be read', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.modes, failure('refactory/console-failed', 'the console exited 2'))
    answer(scripted.strategies, ok(STRATEGIES))
    answer(scripted.packages, ok([PACKAGE]))
    face.load(TAB, signal)
    await settle()
    expect(tab(instance)?.choices).toEqual({ kind: 'failed', message: 'the console exited 2' })
  })

  it('records the reason the strategies could not be read', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.modes, ok([MODE]))
    answer(scripted.strategies, failure('refactory/console-failed', 'no strategies'))
    answer(scripted.packages, ok([PACKAGE]))
    face.load(TAB, signal)
    await settle()
    expect(tab(instance)?.choices).toEqual({ kind: 'failed', message: 'no strategies' })
  })

  it('records the reason the packages could not be read', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.modes, ok([MODE]))
    answer(scripted.strategies, ok(STRATEGIES))
    answer(scripted.packages, failure('refactory/console-failed', 'no registry'))
    face.load(TAB, signal)
    await settle()
    expect(tab(instance)?.choices).toEqual({ kind: 'failed', message: 'no registry' })
  })

  it('does not read a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.load(TAB, controller.signal)
    expect(scripted.modes).not.toHaveBeenCalled()
    expect(scripted.strategies).not.toHaveBeenCalled()
    expect(scripted.packages).not.toHaveBeenCalled()
  })

  it('drops a read a newer one has retired, so the newest answer is the one kept', async () => {
    const { instance, scripted, face, signal } = harness()
    const first = deferred<RemoteResult<readonly RefactoryModeChoice[]>>()
    scripted.modes.mockImplementation(async () => first.promise as never)
    answer(scripted.strategies, ok(STRATEGIES))
    answer(scripted.packages, ok([PACKAGE]))
    face.load(TAB, signal)
    scripted.modes.mockImplementation(async () => ok([AUTHORING]) as never)
    face.reload(TAB, signal)
    await settle()
    first.resolve(ok([MODE]))
    await settle()
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: { ...CHOICES, modes: [AUTHORING] } })
  })

  it('does not re-read a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.reload(TAB, controller.signal)
    expect(scripted.modes).not.toHaveBeenCalled()
  })

  it('marks a reload as in flight, keeping what the tab has chosen', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.modes, ok([MODE]))
    answer(scripted.strategies, ok(STRATEGIES))
    answer(scripted.packages, ok([PACKAGE]))
    face.load(TAB, signal)
    await settle()
    face.choose(TAB, MODE.id)
    face.reload(TAB, signal)
    expect(tab(instance)?.choices).toEqual({ kind: 'loading' })
    expect(tab(instance)?.chosen).toBe(MODE.id)
    expect(scripted.modes).toHaveBeenCalledTimes(2)
  })

  it('chooses a mode and a level-two name, which it only reports', () => {
    const { instance, face } = harness()
    face.choose(TAB, MODE.id)
    face.setKnob(TAB, 'rounds', '7')
    face.setSetting(TAB, 'optimizer', 'random')
    expect(tab(instance)?.chosen).toBe(MODE.id)
    expect(tab(instance)?.knobs).toEqual({ rounds: '7' })
    expect(tab(instance)?.settings).toEqual({ ...NO_SETTINGS, optimizer: 'random' })
  })
})

describe('refactoryFace — starting a run', () => {
  it('releases a pending launch after Watch changes without taking over the selected run', async () => {
    const { instance, scripted, face, signal } = harness()
    const receipt = deferred<RemoteResult<typeof RECEIPT>>()
    scripted.start.mockImplementation(async () => receipt.promise)
    scripted.status.mockImplementation(async (run: string) => ok({ ...FRAME, root: run }))
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    face.watch(TAB, '/runs/other', signal)
    await settle()
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    expect(scripted.start).toHaveBeenCalledTimes(1)
    receipt.resolve(ok(RECEIPT))
    await settle()
    expect(tab(instance)?.watching).toBe('/runs/other')
    expect(tab(instance)?.starting).toBe(false)
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    expect(scripted.start).toHaveBeenCalledTimes(2)
    await settle()
  })

  it('accepts one launch until the receipt settles, then permits retry', async () => {
    const { instance, scripted, face, signal } = harness()
    const receipt = deferred<RemoteResult<typeof RECEIPT>>()
    scripted.start.mockImplementation(async () => receipt.promise)
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    expect(scripted.start).toHaveBeenCalledTimes(1)
    expect(tab(instance)?.starting).toBe(true)
    receipt.resolve(failure('refactory/console-failed', 'try again'))
    await settle()
    expect(tab(instance)?.starting).toBe(false)
    face.start(TAB, MODE.id, [], NO_SETTINGS, signal)
    expect(scripted.start).toHaveBeenCalledTimes(2)
    await settle()
  })

  it('starts the chosen mode with only the settings that were chosen, its knobs, then reads the run', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.start, ok(RECEIPT))
    answer(scripted.status, ok(FRAME))
    face.start(
      TAB,
      MODE.id,
      [{ name: 'rounds', value: '4' }, { name: 'source_root', value: '/tmp/pkg' }],
      { ...NO_SETTINGS, optimizer: 'random' },
      signal,
    )
    await settle()
    expect(scripted.start).toHaveBeenCalledWith({
      mode: 'p/adaptive',
      settings: { optimizer: 'random' },
      knobs: ['rounds=4', 'source_root=/tmp/pkg'],
    }, signal)
    expect(tab(instance)?.watching).toBe('/runs/live')
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 0, signal)
    expect(tab(instance)?.frame).toEqual({ kind: 'ready', value: FRAME })
  })

  it('takes the whole settings object when every seam was chosen', async () => {
    const { scripted, face, signal } = harness()
    answer(scripted.start, ok(RECEIPT))
    answer(scripted.status, ok(FRAME))
    face.start(TAB, MODE.id, [], { strategy: 'beam', optimizer: 'random', evaluator: 'score' }, signal)
    await settle()
    expect(scripted.start).toHaveBeenCalledWith({
      mode: 'p/adaptive',
      settings: { strategy: 'beam', optimizer: 'random', evaluator: 'score' },
      knobs: [],
    }, signal)
  })

  it('shows the Host\'s refusal instead of starting a run', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.start, failure('refactory/authoring-mode', '"p/create" is an authoring mode'))
    face.start(TAB, AUTHORING.id, [], NO_SETTINGS, signal)
    await settle()
    expect(tab(instance)?.notice).toEqual({
      code: 'refactory/authoring-mode',
      message: '"p/create" is an authoring mode',
    })
    expect(tab(instance)?.watching).toBeNull()
    expect(scripted.status).not.toHaveBeenCalled()
  })

  it('does not start a run for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.start(TAB, MODE.id, [], NO_SETTINGS, controller.signal)
    expect(scripted.start).not.toHaveBeenCalled()
  })
})

describe('refactoryFace — using one mode once', () => {
  it.each(['before launch', 'after launch'] as const)('keeps the selected run and its gates when Watch changes %s', async (timing) => {
    const { instance, scripted, face, signal } = harness()
    const prepared = deferred<void>()
    const completed = deferred<void>()
    scripted.remote.refactoryConsole.runStream = async function* () {
      await prepared.promise
      yield { kind: 'launch', launch: { mode: MODE.id, root: SINGLE.root, phase: 'prepared' } }
      await completed.promise
      yield { kind: 'result', run: SINGLE }
    }
    scripted.status.mockImplementation(async (run: string) => ok({ ...FRAME, root: run }))
    scripted.steer.mockImplementation(async ({ run }: { run: string }) => ok({
      ...GATED_STEERING, run, gates: [{ id: 'approve', prompt: `Approve only ${run}`, verdicts: ['proceed'] }],
    }))
    face.run(TAB, MODE.id, [], signal)
    if (timing === 'after launch') {
      prepared.resolve()
      await settle()
      expect(tab(instance)?.watching).toBe(SINGLE.root)
    }
    face.watch(TAB, '/runs/other', signal)
    await settle()
    prepared.resolve()
    await settle()
    expect(tab(instance)?.watching).toBe('/runs/other')
    completed.resolve()
    await settle()
    await settle()
    expect(tab(instance)?.single).toEqual({ kind: 'ready', value: SINGLE })
    expect(tab(instance)?.watching).toBe('/runs/other')
    expect(tab(instance)?.frame).toMatchObject({ kind: 'ready', value: { root: '/runs/other' } })
    expect(tab(instance)?.steering).toMatchObject({ kind: 'ready', value: {
      run: '/runs/other', gates: [{ id: 'approve', prompt: 'Approve only /runs/other' }],
    } })
    expect(signal.aborted).toBe(false)
  })

  it('uses the chosen mode once with the same knob values, and keeps the result', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.run, ok(SINGLE))
    face.run(TAB, MODE.id, [{ name: 'rounds', value: '4' }, { name: 'source_root', value: '/tmp/pkg' }], signal)
    expect(tab(instance)?.single).toEqual({ kind: 'running' })
    await settle()
    expect(scripted.run).toHaveBeenCalledWith({
      mode: 'p/adaptive',
      knobs: ['rounds=4', 'source_root=/tmp/pkg'],
    }, signal)
    expect(tab(instance)?.single).toEqual({ kind: 'ready', value: SINGLE })
  })

  it('keeps a run that stopped at a gate, so the body can say a person is being waited on', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.run, ok(GATED))
    face.run(TAB, MODE.id, [], signal)
    await settle()
    expect(tab(instance)?.single).toEqual({ kind: 'ready', value: GATED })
  })

  it('shows the Host\'s refusal, and stops waiting on a run that will not arrive', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.run, failure('refactory/authoring-mode', '"p/create" is an authoring mode'))
    face.run(TAB, AUTHORING.id, [], signal)
    await settle()
    expect(tab(instance)?.notice).toEqual({
      code: 'refactory/console-failed',
      message: '"p/create" is an authoring mode',
    })
    expect(tab(instance)?.single).toBeNull()
  })

  it('does not use a mode for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.run(TAB, MODE.id, [], controller.signal)
    expect(scripted.run).not.toHaveBeenCalled()
  })
})

describe('refactoryFace — reading and stopping a run', () => {
  it('reads one status frame from the offset the reader last reached', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, ok(FRAME))
    face.refresh(TAB, '/runs/live', 3, signal)
    expect(tab(instance)?.frame).toEqual({ kind: 'loading' })
    await settle()
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 3, signal)
    expect(tab(instance)?.frame).toEqual({ kind: 'ready', value: FRAME })
  })

  it('shows why a status could not be read', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, failure('refactory/console-failed', '/runs/gone is not a run directory'))
    face.refresh(TAB, '/runs/gone', 0, signal)
    await settle()
    // The code travels with the message so the board can tell a payload the
    // collection cap cut from a console that printed no JSON (R-5.4).
    expect(tab(instance)?.frame).toEqual({
      kind: 'failed', code: 'refactory/console-failed', message: '/runs/gone is not a run directory',
    })
  })

  it('does not read a status for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.refresh(TAB, '/runs/live', 0, controller.signal)
    expect(scripted.status).not.toHaveBeenCalled()
  })

  it('asks the run to stop, then reads where it got to', async () => {
    const { scripted, face, signal } = harness()
    answer(scripted.stop, ok({ stopped: true }))
    answer(scripted.status, ok(FRAME))
    face.stop(TAB, '/runs/live', signal)
    await settle()
    expect(scripted.stop).toHaveBeenCalledWith('/runs/live', signal)
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 0, signal)
  })

  it('shows why a run could not be asked to stop', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.stop, failure('refactory/console-failed', 'the run would not stop'))
    face.stop(TAB, '/runs/live', signal)
    await settle()
    expect(tab(instance)?.notice).toEqual({ code: 'refactory/console-failed', message: 'the run would not stop' })
    expect(scripted.status).not.toHaveBeenCalled()
  })

  it('does not ask a run to stop for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.stop(TAB, '/runs/live', controller.signal)
    expect(scripted.stop).not.toHaveBeenCalled()
  })
})

describe('refactoryFace — the watched run\'s ledger', () => {
  it('reads the ledger and the gates the run is stopped at', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, ok(GATED_STEERING))
    face.readSteering(TAB, '/runs/live', signal)
    expect(tab(instance)?.steering).toEqual({ kind: 'loading' })
    await settle()
    expect(scripted.steer).toHaveBeenCalledWith({ run: '/runs/live' }, signal)
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: GATED_STEERING })
  })

  it('records why the ledger could not be read, where the ledger is drawn', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, failure('refactory/console-failed', 'no ledger at /runs/live'))
    face.readSteering(TAB, '/runs/live', signal)
    await settle()
    expect(tab(instance)?.steering).toEqual({ kind: 'failed', message: 'no ledger at /runs/live' })
    expect(tab(instance)?.notice).toBeNull()
  })

  it('appends a note and takes the ledger the Host answers with', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, ok(STEERED))
    face.addNote(TAB, '/runs/live', 'try the other seed', 'correction', signal)
    await settle()
    expect(scripted.steer).toHaveBeenCalledWith(
      { run: '/runs/live', note: 'try the other seed', kind: 'correction' },
      signal,
    )
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: STEERED })
  })

  it('records a verdict for the gate the run is stopped at', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, ok(GATED_STEERING))
    face.answerGate(TAB, '/runs/live', 'approve', 'proceed', signal)
    await settle()
    expect(scripted.steer).toHaveBeenCalledWith({ run: '/runs/live', verdicts: ['approve=proceed'] }, signal)
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: GATED_STEERING })
  })

  it('controls the run itself rather than its ledger', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, ok(STEERED))
    face.controlRun(TAB, '/runs/live', { control: 'pause' }, signal)
    await settle()
    expect(scripted.steer).toHaveBeenCalledWith({ run: '/runs/live', control: 'pause' }, signal)
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: STEERED })
  })

  it('carries the candidate a keep or a reject names, and the name an adopt writes', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.steer, ok(STEERED))
    face.controlRun(TAB, '/runs/live', { control: 'reject', target: 'bbbb22', lineage: true }, signal)
    await settle()
    expect(scripted.steer).toHaveBeenCalledWith(
      { run: '/runs/live', control: 'reject', target: 'bbbb22', lineage: true },
      signal,
    )

    answer(scripted.steer, ok(STEERED))
    face.controlRun(TAB, '/runs/live', { control: 'adopt', as: 'better' }, signal)
    await settle()
    expect(scripted.steer).toHaveBeenLastCalledWith(
      { run: '/runs/live', control: 'adopt', as: 'better' },
      signal,
    )
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: STEERED })
  })

  it('reads the candidates, and reads them again after an act that changes them', async () => {
    const { instance, scripted, face, signal } = harness()
    face.readCandidates(TAB, '/runs/live', signal)
    await settle()
    expect(scripted.leaderboard).toHaveBeenCalledWith('/runs/live', signal)
    expect(tab(instance)?.candidates).toEqual({ kind: 'ready', value: CANDIDATES })

    answer(scripted.steer, ok(STEERED))
    scripted.leaderboard.mockClear()
    face.controlRun(TAB, '/runs/live', { control: 'keep', target: 'aaaa11' }, signal)
    await settle()
    expect(scripted.leaderboard).toHaveBeenCalledWith('/runs/live', signal)
  })

  it('records why the candidates could not be read, where the table is drawn', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.leaderboard, failure('refactory/console-failed', 'no such run'))
    face.readCandidates(TAB, '/runs/gone', signal)
    await settle()
    expect(tab(instance)?.candidates).toEqual({
      kind: 'failed',
      message: 'no such run',
    })
  })

  it('shows a refused note as the notice, keeping the ledger that was already read', async () => {
    const { instance, scripted, face, signal } = harness()
    instance.actions.steered(TAB, STEERING)
    answer(scripted.steer, failure('refactory/console-failed', 'an empty note records nothing'))
    face.addNote(TAB, '/runs/live', '', 'info', signal)
    await settle()
    expect(tab(instance)?.notice).toEqual({
      code: 'refactory/console-failed',
      message: 'an empty note records nothing',
    })
    expect(tab(instance)?.steering).toEqual({ kind: 'ready', value: STEERING })
  })

  it('refreshes the ledger alongside the status, so a note added elsewhere appears', async () => {
    const { scripted, face, signal } = harness()
    answer(scripted.status, ok(FRAME))
    answer(scripted.steer, ok(STEERING))
    face.refresh(TAB, '/runs/live', 3, signal)
    await settle()
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 3, signal)
    expect(scripted.steer).toHaveBeenCalledWith({ run: '/runs/live' }, signal)
  })

  it('does not read or write a ledger for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.readSteering(TAB, '/runs/live', controller.signal)
    face.addNote(TAB, '/runs/live', 'no', 'info', controller.signal)
    face.answerGate(TAB, '/runs/live', 'approve', 'proceed', controller.signal)
    expect(scripted.steer).not.toHaveBeenCalled()
  })
})

describe('refactoryFace — the journal history the panel holds', () => {
  it('reads page 0 on attach, and every later read continues from the highest held cursor', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(0, 200, 450) }))
    face.watch(TAB, '/runs/live', signal)
    await settle()
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 0, signal)
    expect(tab(instance)?.activity).toMatchObject({ through: 200, total: 450, complete: false, paged: true })
    expect(tab(instance)?.activity?.events).toHaveLength(200)

    // A poll advances the journal where the held history ends; it never resets
    // to 0 (R-4.2, F-2).
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(200, 200, 450) }))
    face.refresh(TAB, '/runs/live', FRAME.cursor, signal, { after: 200 })
    await settle()
    expect(scripted.status).toHaveBeenLastCalledWith('/runs/live', 200, signal)
    expect(tab(instance)?.activity).toMatchObject({ through: 400, total: 450, complete: false })
    expect(tab(instance)?.activity?.events).toHaveLength(400)

    // The answering page that reports `has_more: false` is the completeness
    // evidence the held history records (R-2.2).
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(400, 200, 450) }))
    face.refresh(TAB, '/runs/live', FRAME.cursor, signal, { after: 400 })
    await settle()
    expect(tab(instance)?.activity).toMatchObject({ through: 450, total: 450, complete: true })
    expect(tab(instance)?.activity?.events).toHaveLength(450)
  })

  it('starts the live stream where the held history ends, not at 0', async () => {
    const { instance, scripted, face, signal } = harness()
    const asked: number[] = []
    const frames: unknown[] = []
    scripted.remote.refactoryConsole.watchStream = async function* (_run, after) {
      asked.push(after)
      yield { status: { ...FRAME, activity: ACTIVITY(200, 200, 450) }, steering: STEERING }
    }
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(0, 200, 450) }))
    face.watch(TAB, '/runs/live', signal)
    // The stream opens once the first answer names the journal cursor it starts
    // from: the stream owns that cursor and only advances, so starting it at 0
    // would make every poll re-read the journal from the top (F-2).
    await vi.waitFor(() => { expect(asked).toEqual([200]) })
    // The streamed page is spliced onto the held history, and the summary is
    // the stream's own.
    expect(tab(instance)?.activity).toMatchObject({ through: 400 })
    expect(tab(instance)?.frame).toMatchObject({ kind: 'ready' })
    void frames
  })

  it('exhausts the journal from 0 to has_more: false when verbose is chosen', async () => {
    const { instance, scripted, face, signal } = harness()
    scripted.status.mockImplementation(async (_run: string, after: number) =>
      ok({ ...FRAME, activity: ACTIVITY(after, 200, 450) }) as never)
    face.watch(TAB, '/runs/live', signal)
    await settle()
    scripted.status.mockClear()
    face.setActivityLevel(TAB, '/runs/live', 'verbose', signal)
    await settle()
    await settle()
    await settle()
    // From 0, one page per answer, until the answer that says there is no more.
    expect(scripted.status.mock.calls.map(call => call[1])).toEqual([0, 200, 400])
    expect(tab(instance)?.activityLevel).toBe('verbose')
    expect(tab(instance)?.activity).toMatchObject({ through: 450, total: 450, complete: true })
    expect(tab(instance)?.activity?.events.map(event => event.at)).toEqual(
      Array.from({ length: 450 }, (_unused, index) => `t${index}`),
    )
  })

  it('does not fetch for compact or standard, which only change what is drawn', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(0, 200, 450) }))
    face.watch(TAB, '/runs/live', signal)
    await settle()
    scripted.status.mockClear()
    face.setActivityLevel(TAB, '/runs/live', 'compact', signal)
    face.setActivityLevel(TAB, '/runs/live', 'standard', signal)
    await settle()
    expect(scripted.status).not.toHaveBeenCalled()
    expect(tab(instance)?.activityLevel).toBe('standard')
    expect(tab(instance)?.activity?.events).toHaveLength(200)
  })

  it('loads the page before the oldest held record, one press at a time', async () => {
    const { instance, scripted, face, signal } = harness()
    scripted.status.mockImplementation(async (_run: string, after: number) =>
      ok({ ...FRAME, activity: ACTIVITY(after, 200, 450) }) as never)
    face.watch(TAB, '/runs/live', signal)
    await settle()
    face.refresh(TAB, '/runs/live', FRAME.cursor, signal, { after: 200 })
    await settle()
    expect(tab(instance)?.activity?.events).toHaveLength(400)
    scripted.status.mockClear()
    face.loadEarlierActivity(TAB, '/runs/live', 200, signal)
    await settle()
    // The previous page ending at the oldest held index (200) starts at 0.
    expect(scripted.status).toHaveBeenCalledWith('/runs/live', 0, signal)
    expect(tab(instance)?.activity?.events.map(event => event.at)).toEqual(
      Array.from({ length: 400 }, (_unused, index) => `t${index}`),
    )
  })

  it('resets both cursors and drops the held history when another run is watched', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(0, 200, 450) }))
    face.watch(TAB, '/runs/live', signal)
    await settle()
    expect(tab(instance)?.activity?.events).toHaveLength(200)
    answer(scripted.status, ok({ ...FRAME, root: '/runs/other', activity: ACTIVITY(0, 200, 60, '/runs/other') }))
    face.watch(TAB, '/runs/other', signal)
    await settle()
    // R-4.4: page 0 of the new run, and none of the old run's records.
    expect(scripted.status).toHaveBeenLastCalledWith('/runs/other', 0, signal)
    expect(tab(instance)?.activity?.events).toHaveLength(60)
    expect(tab(instance)?.activity).toMatchObject({ through: 60, total: 60, complete: true })
  })

  it('treats a frame whose activity names another run as a gap rather than splicing it', async () => {
    const { instance, scripted, face, signal } = harness()
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(0, 200, 450) }))
    face.watch(TAB, '/runs/live', signal)
    await settle()
    // A frame for a different journal is not this history's records (R-4.5).
    answer(scripted.status, ok({ ...FRAME, activity: ACTIVITY(200, 200, 450, '/runs/other') }))
    face.refresh(TAB, '/runs/live', FRAME.cursor, signal, { after: 200 })
    await settle()
    expect(tab(instance)?.activity?.events).toHaveLength(200)
    expect(tab(instance)?.activity).toMatchObject({ through: 200 })
  })

  it('shows an older console\'s answer as unpaged rather than as the whole history', async () => {
    const { instance, scripted, face, signal } = harness()
    const { has_more: _more, total: _total, after: _after, ...unpaged } = ACTIVITY(0, 3, 3)
    answer(scripted.status, ok({ ...FRAME, activity: { ...unpaged, paged: false } }))
    face.watch(TAB, '/runs/live', signal)
    await settle()
    // R-6.3: the whole `events` it received, marked as a console that does not page.
    expect(tab(instance)?.activity).toMatchObject({ paged: false, events: unpaged.events })
  })
})

describe('refactoryFace — changing a package', () => {
  it('disables a package over the whole change, then takes the catalogue that came back', async () => {
    const { instance, scripted, face, signal } = harness()
    instance.actions.loaded(TAB, CHOICES)
    answer(scripted.setPackageEnabled, ok([{ ...PACKAGE, enabled: false }]))
    answer(scripted.modes, ok(CHOICES.modes))
    answer(scripted.strategies, ok(CHOICES.strategies))
    answer(scripted.packages, ok([{ ...PACKAGE, enabled: false }]))
    face.setEnabled(TAB, 'p', false, signal)
    expect(tab(instance)?.busy).toBe(true)
    await settle()
    expect(scripted.setPackageEnabled).toHaveBeenCalledWith('p', false, signal)
    expect(tab(instance)?.busy).toBe(false)
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: { ...CHOICES, packages: [{ ...PACKAGE, enabled: false }] } })
  })

  it('removes one package, and imports one under the name it was given', async () => {
    const { instance, scripted, face, signal } = harness()
    instance.actions.loaded(TAB, CHOICES)
    answer(scripted.removePackage, ok([]))
    answer(scripted.importPackage, ok([PACKAGE]))
    answer(scripted.modes, ok(CHOICES.modes))
    answer(scripted.strategies, ok(CHOICES.strategies))
    answer(scripted.packages, ok([]))
    face.remove(TAB, 'p', signal)
    await settle()
    expect(scripted.removePackage).toHaveBeenCalledWith('p', signal)
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: { ...CHOICES, packages: [] } })
    answer(scripted.packages, ok([PACKAGE]))
    face.importPackage(TAB, '/src/pack', 'renamed', signal)
    await settle()
    expect(scripted.importPackage).toHaveBeenCalledWith('/src/pack', 'renamed', signal)
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: CHOICES })
  })

  it('shows why a package change was refused, and stops waiting on it', async () => {
    const { instance, scripted, face, signal } = harness()
    instance.actions.loaded(TAB, CHOICES)
    answer(scripted.removePackage, failure('refactory/console-failed', 'shipped packages are read-only'))
    face.remove(TAB, 'p', signal)
    await settle()
    expect(tab(instance)?.notice).toEqual({ code: 'refactory/console-failed', message: 'shipped packages are read-only' })
    expect(tab(instance)?.busy).toBe(false)
  })

  it('does not change a package for a tab whose record has already ended', () => {
    const { scripted, face, controller } = harness()
    controller.abort()
    face.setEnabled(TAB, 'p', true, controller.signal)
    face.remove(TAB, 'p', controller.signal)
    face.importPackage(TAB, '/src/pack', '', controller.signal)
    expect(scripted.setPackageEnabled).not.toHaveBeenCalled()
    expect(scripted.removePackage).not.toHaveBeenCalled()
    expect(scripted.importPackage).not.toHaveBeenCalled()
  })

  it('lets the newest of two overlapping changes write the tab', async () => {
    const { instance, scripted, face, signal } = harness()
    instance.actions.loaded(TAB, CHOICES)
    const first = deferred<RemoteResult<readonly RefactoryPackageChoice[]>>()
    scripted.removePackage.mockImplementation(async () => first.promise as never)
    answer(scripted.importPackage, ok([PACKAGE]))
    answer(scripted.modes, ok(CHOICES.modes))
    answer(scripted.strategies, ok(CHOICES.strategies))
    answer(scripted.packages, ok([PACKAGE]))
    face.remove(TAB, 'p', signal)
    face.importPackage(TAB, '/src/pack', '', signal)
    await settle()
    first.resolve(failure('refactory/console-failed', 'too late'))
    await settle()
    // The retired change neither wrote its refusal nor cleared the flag the
    // second change is still holding.
    expect(tab(instance)?.notice).toBeNull()
    expect(tab(instance)?.choices).toEqual({ kind: 'ready', value: CHOICES })
  })
})
