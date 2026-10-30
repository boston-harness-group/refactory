/**
 * The picker's asynchronous half: reading the described choices and the run's
 * status into the store, and dispatching the gestures that change the Host.
 *
 * Most gestures return immediately. The note form awaits confirmation before
 * clearing its draft. The component calls `load` / `choose` / `setKnob` /
 * `setSetting` / `start` / `run` / `refresh` / `stop` and the package gestures, and
 * this face performs the Remote call and writes the outcome through the store's own
 * actions — the Slot-standard `inject` face, so the write set stays the store's and
 * the body only decides what to draw. `start` and `run` differ in one thing: `start`
 * dispatches the outer loop and then reads its frames, while `run` uses the mode
 * once and takes the single answer the console returns.
 *
 * One read has one settlement in force: asking again — the reload gesture, a
 * refresh while one is in flight — retires the read still outstanding for that
 * tab, whose settlement then writes nothing. Cleanup rides the owner's `signal`:
 * no request is made for a record that already ended, and when the record goes
 * away the bucket and this tab's bookkeeping are forgotten, so no later
 * settlement writes to it.
 *
 * A Remote call does not reject: the result carries the failure, which is stored
 * as the Host's code and message and turned into copy where the copy lives.
 */
import type { RemoteFailure, RemoteResult } from '@deepseek-ai/dsh-api-remotes/client'
import type { ClientConnectionRpc } from '@deepseek-ai/dsh-client-connection/client'
import type { RefactoryConsoleRemote, RefactoryNoteKind, RefactoryPackageChoice } from '@deepseek-ai/dsh-refactory-console/types'
import type { BoundActions } from '@deepseek-ai/dsh-client-store'
import type { TabId } from '@deepseek-ai/dsh-client-ui-dockkit'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { RefactoryActivity, RefactoryRunStatus } from '@deepseek-ai/dsh-refactory-console/types'
import type {
  RefactoryActivityLevel,
  RefactoryActivityPage,
  RefactoryKnobSetting,
  RefactorySettingName,
  RefactorySettings,
  createRefactoryStore,
} from './store.ts'
import { searchBudget } from './store.ts'

/** The slice of the Client Remote face this package calls: the `refactoryConsole` namespace. */
export type RefactoryRemote = {
  readonly refactoryConsole: Pick<
    RefactoryConsoleRemote,
    'help' | 'workers' | 'project' | 'projects' | 'modes' | 'strategies' | 'packages' | 'start' | 'run' | 'runStream' | 'status' | 'leaderboard'
      | 'runs' | 'graphControl' | 'watchStream' | 'steer' | 'stop' | 'setPackageEnabled' | 'removePackage' | 'importPackage'
      | 'profiles' | 'profileSave' | 'profileResolve' | 'comparison'
      | 'steer' | 'stop' | 'setPackageEnabled' | 'removePackage' | 'importPackage'
  >
}

/**
 * Bind the `refactoryConsole` namespace to the Connection's generic RPC.
 *
 * The Host service is discovered by the gateway from its `@Remote` markers, and
 * the gateway reads each argument by the method's declared parameter name. The
 * pinned Client Remote assembly is a prebuilt bundle carrying the harness's own
 * namespaces, so it mounts no `remote.refactoryConsole` service for this fork's
 * namespace; the same `/api` channel answers `refactoryConsole/<method>`, so this
 * sends the named arguments the SRC descriptor reads.
 * @param rpc - the Client Connection's generic unary RPC caller.
 * @param runStream - the mounted Gateway stream; absent retains the unary adapter contract.
 * @returns the namespace face the picker calls.
 */
export function refactoryConsoleOverConnection(
  rpc: ClientConnectionRpc,
  runStream?: RefactoryConsoleRemote['runStream'],
  watchStream?: RefactoryConsoleRemote['watchStream'],
): RefactoryRemote {
  const call = <T>(
    method: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<RemoteResult<T>> =>
    rpc.call('/api', `refactoryConsole/${method}`, { args }, signal) as Promise<RemoteResult<T>>
  return {
    refactoryConsole: {
      ...watchStream === undefined ? {} : { watchStream },
      runs: (signal) => call('runs', {}, signal),
      graphControl: (request, signal) => call('graphControl', { request }, signal),
      profiles: (signal) => call('profiles', {}, signal),
      profileSave: (name, spec, signal) => call('profileSave', { name, spec }, signal),
      profileResolve: (request, signal) => call('profileResolve', { request }, signal),
      comparison: (run, signal) => call('comparison', { run }, signal),
      project: (signal) => call('project', {}, signal),
      modes: (signal) => call('modes', {}, signal),
      strategies: (signal) => call('strategies', {}, signal),
      packages: (signal) => call('packages', {}, signal),
      start: (request, signal) => call('start', { request }, signal),
      run: (request, signal) => call('run', { request }, signal),
      runStream: runStream ?? (async function* (request, signal) {
        const result = await call<import('@deepseek-ai/dsh-refactory-console/types').RefactorySingleRun>('run', { request }, signal)
        if (!result.ok) throw new Error(result.error.message)
        yield { kind: 'result' as const, run: result.value }
      }),
      help: (topic, signal) => call('help', { topic }, signal),
      workers: signal => call('workers', {}, signal),
      status: (run, after, signal) => call('status', { run, after }, signal),
      leaderboard: (run, signal) => call('leaderboard', { run }, signal),
      steer: (request, signal) => call('steer', { request }, signal),
      stop: (run, signal) => call('stop', { run }, signal),
      setPackageEnabled: (name, enabled, signal) => call('setPackageEnabled', { name, enabled }, signal),
      removePackage: (name, signal) => call('removePackage', { name }, signal),
      importPackage: (source, name, signal) => call('importPackage', { source, name }, signal),
    },
  }
}

/**
 * The knob values a gesture sends, as the console's repeatable `--knob NAME=VALUE`.
 * @param values - one value per declared knob, in declaration order.
 * @returns one `NAME=VALUE` argument per knob.
 */
function knobArguments(values: readonly RefactoryKnobSetting[]): readonly string[] {
  return values.map(({ name, value }) => `${name}=${value}`)
}

/**
 * One control act as the body asks for it: the verb, and the candidate or package
 * it names.
 *
 * The act travels as one value rather than as four arguments because the console
 * refuses a request that names nothing, and a shape that cannot express "keep,
 * with no candidate" is one the body cannot ask for by accident.
 */
export interface RefactoryControlAct {
  /** The verb the loop honors at its next generation boundary. */
  readonly control: 'pause' | 'resume' | 'stop' | 'keep' | 'reject' | 'adopt'
  /** The candidate `keep` and `reject` name, as the leaderboard lists it. */
  readonly target?: string
  /** With `keep` or `reject`, act on the candidate's descendants too. */
  readonly lineage?: boolean
  /** With `adopt`, the package name to write. */
  readonly as?: string
}

/** The console's default page of journal records: what one read may carry. */
const ACTIVITY_PAGE = 200

/**
 * How long the live view waits for the first status read to name the stream's
 * starting cursor before opening the stream from 0 instead. It matches the budget
 * the previous fixed tick loop implied (200 × 5 ms), but it bounds only a read
 * that never settles: a read that answers settles the anchor immediately.
 */
const ANCHOR_WAIT_MS = 1000

/** One range of a run's journal to read, as the panel asks for it. */
export interface RefactoryActivityWindow {
  /** Journal index the page starts at; `0` when `before` is given. */
  readonly after?: number
  /** Journal index the page ends at: a backwards read for "Load earlier activity". */
  readonly before?: number
}

/** The picker's injected business face, as the body receives it. */
export interface RefactoryInjected {
  readonly listRuns?: (signal: AbortSignal) => Promise<readonly import('@deepseek-ai/dsh-refactory-console/types').RefactoryRunSummary[]>
  readonly graphControl?: (request: { run: string; action: string; node?: string; text?: string; revision?: string }, signal: AbortSignal) => Promise<Record<string, unknown>>

  /** Direct profile operations keep the saved recipe separate from the watched run. */
  readonly discoveryApi?: Pick<RefactoryConsoleRemote, 'help' | 'workers'>
  readonly profileApi?: Pick<RefactoryConsoleRemote, 'profiles' | 'profileSave' | 'profileResolve' | 'comparison'>

  /**
   * Seed this tab's picker and read the described choices into it.
   * @param tabId - the tab being drawn.
   * @param signal - the tab record's lifetime.
   */
  readonly load: (tabId: TabId, signal: AbortSignal) => void
  /**
   * Read the described choices again, keeping what this tab has chosen.
   * @param tabId - the tab being drawn.
   * @param signal - the tab record's lifetime.
   */
  readonly reload: (tabId: TabId, signal: AbortSignal) => void
  /**
   * Choose one of the described modes, or nothing.
   * @param tabId - the tab being drawn.
   * @param id - the chosen catalogue id, or `null`.
   */
  readonly choosePackage: (tabId: TabId, name: string | null) => void
  readonly chooseIntent: (tabId: TabId, intent: 'run' | 'optimize') => void
  readonly choose: (tabId: TabId, id: string | null) => void
  /**
   * Change one of the chosen mode's declared knobs for this run.
   * @param tabId - the tab being drawn.
   * @param name - the knob's declared name.
   * @param value - the value to send, as text.
   */
  readonly setKnob: (tabId: TabId, name: string, value: string) => void
  /**
   * Choose one of the described level-two names.
   * @param tabId - the tab being drawn.
   * @param name - which seam the choice is for.
   * @param value - the chosen name, or the empty string for the default.
   */
  readonly setSetting: (tabId: TabId, name: RefactorySettingName, value: string) => void
  /**
   * Start a run from the chosen mode, its knobs and its settings, then read its
   * first frame.
   * @param tabId - the tab being drawn.
   * @param mode - the chosen mode's catalogue id.
   * @param knobs - the value of every declared knob, in declaration order.
   * @param settings - the chosen level-two names.
   * @param signal - the tab record's lifetime.
   */
  readonly start: (
    tabId: TabId,
    mode: string,
    knobs: readonly RefactoryKnobSetting[],
    settings: RefactorySettings,
    signal: AbortSignal,
  ) => void
  /**
   * Use the chosen mode once, with its knobs, and take the result it left.
   * @param tabId - the tab being drawn.
   * @param mode - the chosen mode's catalogue id.
   * @param knobs - the value of every declared knob, in declaration order.
   * @param signal - the tab record's lifetime.
   */
  readonly run: (
    tabId: TabId,
    mode: string,
    knobs: readonly RefactoryKnobSetting[],
    signal: AbortSignal,
    settings?: RefactorySettings,
  ) => void
  /**
   * Read one status frame for a run.
   *
   * `window` is the journal range this read asks for, and it is separate from
   * `after` because that one indexes the progress file. It only travels when the
   * face was given a window reader: the Remote `status` method takes no journal
   * index, so a caller that omits it gets exactly the frame that method was
   * written to answer with. Either way the frame's `activity.events` is spliced
   * onto the held history by its own `after`.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param after - how many progress lines this reader has already seen.
   * @param signal - the tab record's lifetime.
   * @param window - the journal range to read, when the face reads journal ranges.
   */
  readonly refresh: (
    tabId: TabId,
    run: string,
    after: number,
    signal: AbortSignal,
    window?: RefactoryActivityWindow,
  ) => void
  /** Attach an existing run by the exact directory the CEO or terminal reported. */
  readonly watch: (tabId: TabId, run: string, signal: AbortSignal) => void
  /**
   * Choose how much journalled history the tab draws, and read the whole of it
   * when the choice is `verbose`.
   *
   * The fetch is part of the gesture rather than of the renderer: exhausting the
   * journal is a sequence of Remote reads, so a person switching level is what
   * starts it, and every page it lands is the store's write (R-2.2, F-4).
   * @param tabId - the tab being drawn.
   * @param run - the run being read.
   * @param level - the level to draw at.
   * @param signal - the tab record's lifetime.
   */
  readonly setActivityLevel: (
    tabId: TabId,
    run: string,
    level: RefactoryActivityLevel,
    signal: AbortSignal,
  ) => void
  /**
   * Ask for the page before the oldest record the tab holds.
   * @param tabId - the tab being drawn.
   * @param run - the run being read.
   * @param oldest - the journal index the held history begins at, as drawn.
   * @param signal - the tab record's lifetime.
   */
  readonly loadEarlierActivity: (tabId: TabId, run: string, oldest: number, signal: AbortSignal) => void
  /**
   * Remember the row the history window is drawn from.
   * @param tabId - the tab being drawn.
   * @param row - the row index of the first drawn record.
   */
  readonly scrollActivity: (tabId: TabId, row: number) => void
  /**
   * Read the watched run's intent ledger and the gates it is stopped at.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param signal - the tab record's lifetime.
   */
  readonly readSteering: (tabId: TabId, run: string, signal: AbortSignal) => void
  /**
   * Read the watched run's candidates, with what a person has done to each.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param signal - the tab record's lifetime.
   */
  readonly readCandidates: (tabId: TabId, run: string, signal: AbortSignal) => void
  /**
   * Control the run: hold it, release it, end it, follow a candidate up, dismiss
   * one, or write its current winner as a package.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param act - the verb the loop honors at its next generation boundary, and
   * what it names.
   * @param signal - the tab record's lifetime.
   */
  readonly controlRun: (
    tabId: TabId,
    run: string,
    act: RefactoryControlAct,
    signal: AbortSignal,
  ) => void
  /**
   * Append one note to the watched run's ledger, and take the ledger it answers with.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param note - what the person said.
   * @param kind - what the note is.
   * @param signal - the tab record's lifetime.
   * @returns whether this submission was confirmed for the current watched run.
   */
  readonly addNote: (tabId: TabId, run: string, note: string, kind: RefactoryNoteKind, signal: AbortSignal) => Promise<boolean>
  /**
   * Record a verdict for a gate the watched run is stopped at, and take the ledger
   * it answers with.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param gate - the gate node the verdict answers.
   * @param verdict - the outcome label the gate routes on.
   * @param signal - the tab record's lifetime.
   */
  readonly answerGate: (tabId: TabId, run: string, gate: string, verdict: string, signal: AbortSignal) => void
  /**
   * Ask a run to stop, then read it again.
   * @param tabId - the tab being drawn.
   * @param run - the run id or directory.
   * @param signal - the tab record's lifetime.
   */
  readonly stop: (tabId: TabId, run: string, signal: AbortSignal) => void
  /**
   * Enable or disable one package, then take the catalogue the Host answers with.
   * @param tabId - the tab being drawn.
   * @param name - the package name.
   * @param enabled - the state to set.
   * @param signal - the tab record's lifetime.
   */
  readonly setEnabled: (tabId: TabId, name: string, enabled: boolean, signal: AbortSignal) => void
  /**
   * Remove one workspace package, then take the catalogue the Host answers with.
   * @param tabId - the tab being drawn.
   * @param name - the package name.
   * @param signal - the tab record's lifetime.
   */
  readonly remove: (tabId: TabId, name: string, signal: AbortSignal) => void
  /**
   * Import a package from a directory or git URL, then take the catalogue the
   * Host answers with.
   * @param tabId - the tab being drawn.
   * @param source - the directory or git URL.
   * @param name - the name to import it as; empty keeps its own.
   * @param signal - the tab record's lifetime.
   */
  readonly importPackage: (tabId: TabId, source: string, name: string, signal: AbortSignal) => void
}

/**
 * Bind the picker's face to one Client Remote face.
 * @param remote - the Client Remote face carrying the `refactoryConsole` namespace.
 * @returns the Slot `inject` factory: session and bound actions in, face out.
 */
export function refactoryFace(
  remote: RefactoryRemote,
): (sessionId: SessionId, actions: BoundActions<ReturnType<typeof createRefactoryStore>>) => RefactoryInjected {
  return (
    _sessionId: SessionId,
    actions: BoundActions<ReturnType<typeof createRefactoryStore>>,
  ): RefactoryInjected => {
    // Only a newer request for the same operation may retire its predecessor.
    // In particular, catalogue/status reads cannot retire a long-running use.
    const catalogueGenerations = new Map<TabId, number>()
    const statusGenerations = new Map<TabId, number>()
    const startGenerations = new Map<TabId, number>()
    const runGenerations = new Map<TabId, number>()
    const packageGenerations = new Map<TabId, number>()
    const stopGenerations = new Map<TabId, number>()
    const watchGenerations = new Map<TabId, number>()
    const starting = new Set<TabId>()
    const running = new Set<TabId>()
    const watchers = new Map<TabId, AbortController>()
    /**
     * The same, for the ledger read: a status refresh and a ledger refresh are two
     * independent reads of one run, and retiring one must not drop the other.
     */
    const steeringGenerations = new Map<TabId, number>()
    // A candidate read and a ledger read are two reads of the same tab and must not
    // retire each other; one control act refreshes both.
    const candidateGenerations = new Map<TabId, number>()
    /**
     * The tab's own journal bookkeeping, alongside the store.
     *
     * Only genuinely transient facts live here: the highest cursor already asked
     * for (so a poll and a page cannot ask for the same index twice), whether a
     * verbose pass is in flight, and which run the level gesture asked to exhaust
     * — because a poll must not start fetching a whole journal nobody asked for.
     * Everything a person sees is written to the store instead.
     */
    const heldThrough = new Map<TabId, number>()
    const pageGenerations = new Map<TabId, number>()
    const exhausting = new Map<TabId, string>()
    /** Monotonic across bucket lifetimes: a late response from a closed tab must
     * not match the first request when that id is used again. */
    let sequence = 0
    const nextGeneration = (generations: Map<TabId, number>, tabId: TabId): number => {
      const generation = ++sequence
      generations.set(tabId, generation)
      return generation
    }
    /**
     * Write one call's outcome, unless a newer read retired it. A refusal becomes
     * the notice the body renders, or the read's own failure line where the read
     * is the thing the reader is looking at; `done` runs for a gesture that must
     * clear its own in-flight state, and `failed` for one that must clear it and
     * still keep the Host's code.
     */
    const settle = <T>(
      generations: Map<TabId, number>,
      tabId: TabId,
      generation: number,
      call: Promise<RemoteResult<T>>,
      landed: (value: T) => void,
      done?: () => void,
      failed?: (failure: RemoteFailure) => void,
    ): void => {
      void call.then((result) => {
        if (generations.get(tabId) !== generation) return
        if (result.ok) landed(result.value)
        else if (failed === undefined) {
          actions.noticed(tabId, { code: result.error.code, message: result.error.message })
        } else failed(result.error)
        done?.()
      })
    }
    /**
     * One answer's journal window, as the store's page shape.
     *
     * A console that does not page is reported as one unpaged answer covering
     * index 0: no `after`/`total`/`has_more`, or `paged: false` because the call
     * only succeeded by dropping the paging options an older console refuses.
     * Saying so is what keeps the panel from showing a page as if it were the
     * whole journal (R-6.3, F-8).
     * @param activity - the answer's `activity` member, or nothing.
     * @returns the page, or `null` when the answer carried no journal at all.
     */
    const pageOf = (activity: RefactoryActivity | null | undefined): RefactoryActivityPage | null => {
      if (activity === null || activity === undefined) return null
      const paged = activity.paged !== false
        && typeof activity.after === 'number' && typeof activity.total === 'number'
        && typeof activity.has_more === 'boolean'
      return {
        after: paged ? activity.after as number : 0,
        cursor: paged ? activity.cursor : activity.events.length,
        events: activity.events,
        total: paged ? activity.total as number : activity.events.length,
        paged,
        run: typeof activity.run_id === 'string' && activity.run_id !== '' ? activity.run_id : null,
      }
    }
    /**
     * Give one answer's journal window to the store, spliced where it belongs.
     *
     * A frame whose `activity.run_id` names another run is dropped rather than
     * spliced: two runs' journal indices do not mean the same thing, so a foreign
     * frame is a gap the run switch already refetched page 0 for (R-4.5).
     * @param tabId - the tab being drawn.
     * @param run - the run the answer belongs to.
     * @param answer - the status answer whose page is to be taken.
     * @param earlier - the answer was asked for the page before the oldest held record.
     */
    const takePage = (
      tabId: TabId,
      run: string,
      answer: RefactoryRunStatus,
      earlier: boolean,
    ): void => {
      const page = pageOf(answer.activity)
      if (page === null) return
      if (page.run !== null && page.run !== run && page.run !== answer.root) return
      if (earlier) {
        // A backwards page extends the history towards 0; it must not move the
        // forward cursor, which is where a poll continues from.
        actions.activityEarlier(tabId, run, page)
        return
      }
      // The forward cursor follows the forward page only: `after + len` is where
      // the next forward read continues (`cursor` may be ahead of it when the
      // console counted records the page did not carry).
      heldThrough.set(tabId, Math.max(heldThrough.get(tabId) ?? 0, page.after + page.events.length))
      actions.activityPage(tabId, run, page)
    }
    /**
     * Follow a verbose pass until an answer reports `has_more: false`.
     *
     * Pages are fetched one at a time and spliced by their own `after`, so the
     * pass resumes from the held cursor instead of restarting: a tab switch or a
     * run switch keeps the records already fetched, and the next gesture
     * continues from there (R-2.3, R-4.4). The answer that says `has_more: false`
     * — and the `total` it carries — is the completeness evidence the store's
     * `complete` flag records.
     * @param tabId - the tab whose level asked for the whole history.
     * @param run - the run to exhaust.
     * @param from - the journal index the pass continues at.
     * @param signal - the tab record's lifetime.
     */
    const exhaust = (tabId: TabId, run: string, from: number, signal: AbortSignal): void => {
      if (signal.aborted || exhausting.has(tabId)) return
      exhausting.set(tabId, run)
      const generation = nextGeneration(pageGenerations, tabId)
      actions.activityPageLoading(tabId, run)
      settle(pageGenerations, tabId, generation, remote.refactoryConsole.status(run, from, signal), (frame) => {
        const page = pageOf(frame.activity)
        if (page === null) return
        takePage(tabId, run, frame, false)
        if (exhausting.get(tabId) !== run || !page.paged) return
        // One page per settlement, so each arrival re-renders the window rather
        // than the history and the tab stays answerable while it pages (R-2.5).
        if (frame.activity?.has_more === true) {
          exhausting.delete(tabId)
          exhaust(tabId, run, page.cursor, signal)
        } else exhausting.delete(tabId)
      }, undefined, (failure) => {
        exhausting.delete(tabId)
        actions.activityPageFailed(tabId, run, failure.message)
      })
    }
    /**
     * Read one run's status, and offer its journal window to the held history.
     *
     * `window` asks for a particular journal range; without one the read is the
     * plain status call the service's method was written for. Either way the
     * answer's page is spliced rather than the frame's summary being trusted to
     * describe the history.
     * @param tabId - the tab being drawn.
     * @param run - the run id or directory.
     * @param after - how many progress lines this reader has already seen.
     * @param window - the journal range to read, or none for the method's default.
     * @param signal - the tab record's lifetime.
     */
    const readStatus = (
      tabId: TabId,
      run: string,
      after: number,
      window: RefactoryActivityWindow | undefined,
      signal: AbortSignal,
      onFrame?: (frame: RefactoryRunStatus) => void,
    ): void => {
      if (signal.aborted) return
      const generation = nextGeneration(statusGenerations, tabId)
      actions.frameLoading(tabId)
      const index = window?.after ?? window?.before
      const call = index === undefined
        ? remote.refactoryConsole.status(run, after, signal)
        : remote.refactoryConsole.status(run, index, signal)
      settle(statusGenerations, tabId, generation, call, (frame) => {
        actions.framed(tabId, frame)
        takePage(tabId, run, frame, false)
        onFrame?.(frame)
      }, undefined, (failure) => { actions.frameFailed(tabId, failure.message, failure.code, failure.details) })
    }

    /** Read one run's intent ledger and gates into the store. */
    const loadSteering = (tabId: TabId, run: string, signal: AbortSignal): void => {
      if (signal.aborted) return
      const generation = nextGeneration(steeringGenerations, tabId)
      actions.steeringLoading(tabId)
      settle(steeringGenerations, tabId, generation, console.steer({ run }, signal), (steering) => {
        actions.steered(tabId, steering)
      }, undefined, (failure) => { actions.steeringFailed(tabId, failure.message) })
    }
    /** Read one run's candidates into the store. */
    const readCandidates = (tabId: TabId, run: string, signal: AbortSignal): void => {
      if (signal.aborted) return
      const generation = nextGeneration(candidateGenerations, tabId)
      actions.candidatesLoading(tabId)
      settle(candidateGenerations, tabId, generation, console.leaderboard(run, signal), (candidates) => {
        actions.candidatesRead(tabId, candidates)
      }, undefined, (failure) => { actions.candidatesFailed(tabId, failure.message) })
    }
    /** Change target without allowing the previous target's late replies to land. */
    const watchRun = (tabId: TabId, run: string, signal: AbortSignal): void => {
      if (signal.aborted || run.trim() === '') return
      nextGeneration(watchGenerations, tabId)
      nextGeneration(candidateGenerations, tabId)
      nextGeneration(stopGenerations, tabId)
      watchers.get(tabId)?.abort()
      // R-4.1/R-4.4: attach is the one read that starts at 0, in both cursors,
      // and the held history it drops is the previous run's (the store does that
      // in `watching`), so no page of one run is ever spliced onto another.
      heldThrough.delete(tabId)
      exhausting.delete(tabId)
      actions.watching(tabId, run.trim())
      // Where the stream may start: the first answer's journal cursor, read
      // below. The stream's cursor is the service's own and only advances, so
      // starting it at 0 would make every poll re-read the journal from the top
      // (F-2) — the read is therefore awaited before the stream opens.
      //
      // The wait is a promise the answer settles, not a poll of a shared cell:
      // a fixed 5 ms × 200 tick loop gave a slow Remote read a 1 s budget and then
      // silently fell back to `from = 0`, which re-reads the whole journal. Here
      // the first settled answer wins, and `ANCHOR_WAIT_MS` only bounds a read
      // that never settles at all.
      let settleAnchor: (through: number) => void = () => {}
      const anchor = new Promise<number>((resolve) => { settleAnchor = resolve })
      readStatus(tabId, run.trim(), 0, {}, signal, (frame) => {
        settleAnchor(typeof frame.activity?.cursor === 'number' ? frame.activity.cursor : 0)
      })
      loadSteering(tabId, run.trim(), signal)
      if (console.watchStream !== undefined) {
        const controller = new AbortController()
        watchers.set(tabId, controller)
        const abort = (): void => controller.abort()
        signal.addEventListener('abort', abort, { once: true })
        void (async () => {
          // The answer settles the anchor; the timeout only bounds a read that
          // never settles. A refusal or an abort leaves this unresolved, and the
          // `0` fallback below is the same one the tick loop used, reached in
          // ANCHOR_WAIT_MS rather than after a fixed number of sleeps.
          const from = await Promise.race([
            anchor,
            new Promise<number>((resolve) => { setTimeout(() => resolve(0), ANCHOR_WAIT_MS) }),
          ])
          if (controller.signal.aborted) return
          try {
            // The stream starts where this tab's held history ends, not at 0: the
            // service advances its own cursor from there, and a reset would ask
            // for records the panel already holds (R-4.2, F-2).
            for await (const frame of console.watchStream!(run.trim(), from, controller.signal)) {
              if (controller.signal.aborted || watchers.get(tabId) !== controller) return
              // Retire any older unary frame before accepting streamed state.
              nextGeneration(statusGenerations, tabId)
              nextGeneration(steeringGenerations, tabId)
              actions.framed(tabId, frame.status)
              actions.steered(tabId, frame.steering)
              takePage(tabId, run.trim(), frame.status, false)
            }
          } catch (error) {
            if (!controller.signal.aborted) actions.noticed(tabId, {
              code: 'refactory/watch-disconnected', message: `Live view disconnected; Watch reconnects without restarting work. ${error instanceof Error ? error.message : String(error)}`,
            })
          } finally { signal.removeEventListener('abort', abort) }
        })()
      }
    }
    /** Take the packages a package gesture answered with. */
    const takePackages = (
      tabId: TabId,
      call: Promise<RemoteResult<readonly RefactoryPackageChoice[]>>,
      signal: AbortSignal,
    ): void => {
      const generation = nextGeneration(packageGenerations, tabId)
      actions.busy(tabId, true)
      settle(packageGenerations, tabId, generation, call, (packages) => {
        actions.packages(tabId, packages)
        read(tabId, signal)
      }, () => { actions.busy(tabId, false) })
    }
    const console = remote.refactoryConsole
    /**
     * One read of the project identity and catalogues, retired by the next read of this tab.
     * Both callers have already refused an ended record; this only reads.
     */
    const read = (tabId: TabId, signal: AbortSignal): void => {
      const generation = nextGeneration(catalogueGenerations, tabId)
      void Promise.all([
        console.project(signal),
        console.modes(signal),
        console.strategies(signal),
        console.packages(signal),
        console.projects ? console.projects(signal) : Promise.resolve({ ok: true as const, value: [] as readonly import('@deepseek-ai/dsh-refactory-console/types').RefactoryProject[] }),
      ]).then(([project, modes, strategies, packages, projects]) => {
        if (catalogueGenerations.get(tabId) !== generation) return
        if (!project.ok) { actions.failed(tabId, project.error.message); return }
        if (!modes.ok) { actions.failed(tabId, modes.error.message); return }
        if (!strategies.ok) { actions.failed(tabId, strategies.error.message); return }
        if (!packages.ok) { actions.failed(tabId, packages.error.message); return }
        // When the projects call fails, the panel still loads with an empty
        // target picker and a console note, rather than failing the whole panel.
        const projectsList = projects.ok ? projects.value : []
        if (!projects.ok) globalThis.console.warn('refactory: projects list unavailable:', projects.error.message)
        actions.loaded(tabId, { project: project.value, modes: modes.value, strategies: strategies.value, packages: packages.value, projects: projectsList })
      })
    }
    return {
      async listRuns(signal) {
        const result = await console.runs(signal)
        if (!result.ok) throw new Error(result.error.message)
        return result.value
      },
      async graphControl(request, signal) {
        if (console.graphControl === undefined) throw new Error('Graph controls unavailable in this runtime')
        const result = await console.graphControl(request, signal)
        if (!result.ok) throw new Error(result.error.message)
        return result.value
      },
      profileApi: console,
      discoveryApi: console,
      load(tabId, signal) {
        if (signal.aborted) return
        actions.start(tabId)
        signal.addEventListener('abort', () => {
          for (const generations of [
            catalogueGenerations, statusGenerations, startGenerations, runGenerations,
            packageGenerations, stopGenerations, steeringGenerations, candidateGenerations, watchGenerations,
          ]) generations.delete(tabId)
          starting.delete(tabId)
          running.delete(tabId)
          heldThrough.delete(tabId)
          exhausting.delete(tabId)
          watchers.get(tabId)?.abort()
          watchers.delete(tabId)
          actions.forget(tabId)
        }, { once: true })
        read(tabId, signal)
      },
      reload(tabId, signal) {
        if (signal.aborted) return
        actions.loading(tabId)
        read(tabId, signal)
      },
      choosePackage(tabId, name) { actions.chosePackage(tabId, name) },
      chooseIntent(tabId, intent) { actions.setupIntent(tabId, intent) },
      choose(tabId, id) {
        actions.chose(tabId, id)
      },
      setKnob(tabId, name, value) {
        actions.knob(tabId, name, value)
      },
      setSetting(tabId, name, value) {
        actions.setting(tabId, name, value)
      },
      start(tabId, mode, knobs, settings, signal) {
        if (signal.aborted || starting.has(tabId)) return
        const budget = searchBudget(settings)
        if (budget === null) {
          actions.noticed(tabId, { code: 'refactory/invalid-search-budget', message: 'Search limits must be positive whole numbers.' })
          return
        }
        // Only the names that were chosen travel; an empty seam takes the
        // console's own default.
        const chosen: Record<string, string | number> = { ...budget }
        if (settings.strategy !== '') chosen.strategy = settings.strategy
        if (settings.optimizer !== '') chosen.optimizer = settings.optimizer
        if (settings.evaluator !== '') chosen.evaluator = settings.evaluator
        if (settings.configuration) chosen.settings_json = settings.configuration
        if (settings.profile) chosen.profile = settings.profile
        if (settings.profile_version) chosen.profile_version = Number(settings.profile_version)
        starting.add(tabId)
        actions.starting(tabId, true)
        const generation = nextGeneration(startGenerations, tabId)
        const watchGeneration = watchGenerations.get(tabId)
        settle(startGenerations, tabId, generation, console.start(
          { mode, settings: chosen, knobs: knobArguments(knobs) },
          signal,
        ), (receipt) => {
          if (watchGenerations.get(tabId) === watchGeneration) watchRun(tabId, receipt.root, signal)
        }, () => { starting.delete(tabId); actions.starting(tabId, false) })
      },
      run(tabId, mode, knobs, signal, settings) {
        if (signal.aborted || running.has(tabId)) return
        running.add(tabId)
        const generation = nextGeneration(runGenerations, tabId)
        const watchGeneration = watchGenerations.get(tabId)
        actions.runLoading(tabId)
        const current = (): boolean => !signal.aborted && runGenerations.get(tabId) === generation
        void (async () => {
          let answered = false
          try {
            for await (const event of console.runStream({ mode, knobs: knobArguments(knobs),
              ...settings?.configuration ? { settings: runtimeSettings(settings.configuration) } : {},
            }, signal)) {
              if (!current()) return
              if (event.kind === 'launch') {
                actions.runLaunched(tabId, event.launch)
                // A later explicit Watch owns the view, even while this launch
                // is still preparing. Its worker/result remain independently owned.
                if (event.launch.phase === 'prepared' && watchGenerations.get(tabId) === watchGeneration) {
                  watchRun(tabId, event.launch.root, signal)
                }
              }
              // The durable watcher owns the ledger. A completed use must not
              // put its gates under another run selected in the meantime.
              else { actions.ran(tabId, event.run); answered = true }
            }
            if (!answered && current()) throw new Error('The run stream ended without a final result; inspect its run directory before retrying.')
          } catch (error) {
            if (!current()) return
            const failure = error instanceof Error ? error.message : String(error)
            actions.noticed(tabId, { code: 'refactory/console-failed', message: failure })
            actions.runFailed(tabId)
          } finally {
            // Release this invocation even when its own signal was cancelled.
            // Record teardown deletes the generation first, so it is never revived.
            if (runGenerations.get(tabId) === generation) {
              running.delete(tabId)
              if (signal.aborted) {
                actions.noticed(tabId, { code: 'gateway/cancelled', message: 'Run once was cancelled; inspect its run directory before retrying.' })
                actions.runFailed(tabId)
              }
            }
          }
        })()
      },
      refresh(tabId, run, after, signal, window) {
        // A refresh continues the journal where the tab's held history ends; the
        // progress cursor it also carries indexes a different file (R-4.2-3).
        readStatus(tabId, run, after, window ?? {}, signal)
        // A ledger is appended to while the run goes, so a refresh of the run is
        // also a refresh of what has been said about it.
        loadSteering(tabId, run, signal)
      },
      watch(tabId, run, signal) {
        watchRun(tabId, run, signal)
      },
      setActivityLevel(tabId, run, level, signal) {
        if (signal.aborted) return
        actions.setActivityLevel(tabId, level)
        // Only the whole-history level fetches; the other two draw what is
        // already held, so switching to them costs no read (R-2.7).
        if (level !== 'verbose') { exhausting.delete(tabId); return }
        // From 0, always — and this is a requirement, not an oversight.
        //
        // R-2.2 makes `verbose` mean the held history contains *every* complete
        // record, accumulated page by page until an answer reports
        // `has_more: false`. Starting from the held watermark would only fetch the
        // tail, so the pass would stop as soon as it reached `total`, never reach
        // index 0, and leave `activity.first` wherever the poll happened to leave
        // it — a level that promises every record would quietly hold a suffix.
        //
        // The cost is real but paid once per gesture: a standard → verbose →
        // standard → verbose round-trip re-fetches from 0 each time. Keeping that
        // cheap belongs in the store, which already splices by each page's own
        // `after` and would need to remember a completed pass per run to skip the
        // walk; R-2.2's guarantee is the contract, and re-reading a journal the
        // console pages in O(page_size) is bounded. Verified by T-18: choosing
        // verbose asks [0, 200, 400] and ends with `first: 0`, `complete: true`.
        exhaust(tabId, run, 0, signal)
      },
      loadEarlierActivity(tabId, run, first, signal) {
        if (signal.aborted) return
        // The renderer hands in where the drawn history begins: it is the one
        // place that knows what the store holds, so this gesture needs no second
        // copy of that fact that could disagree with what a person sees.
        if (!Number.isSafeInteger(first) || first <= 0) return
        // The page before the oldest held record: one page per press, walking to
        // index 0. A descending `after` cannot use the console's advancing seek,
        // so it is asked for as the forward page that ends where the held one
        // begins (R-1.7a, F-6).
        const generation = nextGeneration(pageGenerations, tabId)
        actions.activityPageLoading(tabId, run)
        settle(pageGenerations, tabId, generation,
          remote.refactoryConsole.status(run, Math.max(0, first - ACTIVITY_PAGE), signal), (frame) => {
            takePage(tabId, run, frame, true)
          }, undefined, (failure) => { actions.activityPageFailed(tabId, run, failure.message) })
      },
      scrollActivity(tabId, row) {
        actions.activityScrolled(tabId, row)
      },
      readSteering(tabId, run, signal) {
        loadSteering(tabId, run, signal)
      },
      readCandidates(tabId, run, signal) {
        readCandidates(tabId, run, signal)
      },
      controlRun(tabId, run, act, signal) {
        if (signal.aborted) return
        // Replace the previous gesture's notice before dispatch. A later failure
        // from another operation must not be erased by this control's success.
        actions.noticed(tabId, null)
        const generation = nextGeneration(steeringGenerations, tabId)
        settle(steeringGenerations, tabId, generation, console.steer(
          { run, ...act },
          signal,
        ), (steering) => {
          actions.steered(tabId, steering)
          // A keep or a reject changes the rows themselves, and the ledger read is
          // already current; so the same act refreshes the table it acted on.
          readCandidates(tabId, run, signal)
        }, undefined, (failure) => {
          actions.noticed(tabId, { code: failure.code, message: failure.message })
          // A refused adoption also has a durable receipt. Read it so the
          // previously applied control is not left looking like this result.
          loadSteering(tabId, run, signal)
        })
      },
      async addNote(tabId, run, note, kind, signal) {
        if (signal.aborted) return false
        actions.noticed(tabId, null)
        const generation = nextGeneration(steeringGenerations, tabId)
        try {
          const result = await console.steer({ run, note, kind }, signal)
          if (signal.aborted || steeringGenerations.get(tabId) !== generation) return false
          if (result.ok) { actions.steered(tabId, result.value); return true }
          actions.noticed(tabId, { code: result.error.code, message: result.error.message })
        } catch (error) {
          if (!signal.aborted && steeringGenerations.get(tabId) === generation) {
            actions.noticed(tabId, { code: 'refactory/console-failed', message: error instanceof Error ? error.message : String(error) })
          }
        }
        return false
      },
      answerGate(tabId, run, gate, verdict, signal) {
        if (signal.aborted) return
        const generation = nextGeneration(steeringGenerations, tabId)
        settle(steeringGenerations, tabId, generation, console.steer(
          { run, verdicts: [`${gate}=${verdict}`] },
          signal,
        ), (steering) => {
          actions.steered(tabId, steering)
        }, undefined, (failure) => {
          actions.noticed(tabId, { code: failure.code, message: failure.message })
        })
      },
      stop(tabId, run, signal) {
        if (signal.aborted) return
        const generation = nextGeneration(stopGenerations, tabId)
        settle(stopGenerations, tabId, generation, console.stop(run, signal), () => {
          // Continue the held history: a stop is not a new run (R-4.2).
          readStatus(tabId, run, 0, {}, signal)
        })
      },
      setEnabled(tabId, name, enabled, signal) {
        if (signal.aborted) return
        takePackages(tabId, console.setPackageEnabled(name, enabled, signal), signal)
      },
      remove(tabId, name, signal) {
        if (signal.aborted) return
        takePackages(tabId, console.removePackage(name, signal), signal)
      },
      importPackage(tabId, source, name, signal) {
        if (signal.aborted) return
        takePackages(tabId, console.importPackage(source, name, signal), signal)
      },
    }
  }
}

/** Only worker routing and resources apply when using a mode once. */
export function runtimeSettings(configuration: string): Record<string, string | number> {
  const value = JSON.parse(configuration) as Record<string, unknown>
  const allowed = ['worker_backend', 'provider', 'model', 'effort', 'permission_mode',
    'max_elapsed_s', 'max_tokens', 'validation_reserve_s', 'validation_reserve_tokens']
  const result = Object.fromEntries(Object.entries(value).filter(([key, entry]) => allowed.includes(key)
    && (typeof entry === 'string' || typeof entry === 'number'))) as Record<string, string | number>
  for (const key of ['role_routes', 'node_routes']) if (value[key] && typeof value[key] === 'object') result[key] = JSON.stringify(value[key])
  return result
}
