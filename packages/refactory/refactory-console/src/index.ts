/**
 * The Web picker's Host service: the console, over Typert Remote.
 *
 * The loop is Python and stays there; the tool and the Web view are its two
 * DSH-side readers. This service is the second one: it holds the same `Console`
 * the tool holds, mounts the same deployment schema, and runs the same
 * `python -m refactory_dsh.console` invocations, so a choice a person makes in
 * the panel is the choice the model makes through the tool.
 *
 * What this service adds over a bare call is the picker's two rules:
 *
 * 1. Every choice is described. Modes, strategies and packages come from the
 *    console's own catalogues, so the panel never asks a person to type a name
 *    it could have offered — a mode's declared knobs included, which is what
 *    lets the panel supply an input the search cannot invent.
 * 2. A refusal says what to do instead. `start` and `run` read the catalogue's
 *    `action` field — the console's, not a second derivation — and name the
 *    authoring verb for a mode neither can act on, before shelling out.
 *
 * A run stopped at a gate a person owns is not a refusal: `run` answers with the
 * spine's own word for where the run got to, so the panel can say a person is
 * being waited on rather than showing a failure.
 *
 * Failures are one `RemoteError` per reason: `gateway/bad-request` for an
 * address the picker must supply, `refactory/unknown-mode`,
 * `refactory/authoring-mode` and `refactory/package-disabled` for the three
 * described-choice refusals, `refactory/console-failed` for a console refusal
 * carrying the console's own message and the argv to reproduce it, and
 * `refactory/console-truncated` for a payload the collection cap cut before the
 * console printed JSON — a size problem, not a refusal, and reported as one.
 * @module @deepseek-ai/dsh-refactory-console
 */

import { randomUUID } from 'node:crypto'
import { mkdir, readFile, realpath } from 'node:fs/promises'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type {} from '@deepseek-ai/dsh-subprocess'
import {
  CONSOLE_DEFAULTS, Console, isRecord, registryInvocation, resolveConsoleSettings, runInvocation,
  startInvocation,
} from '@deepseek-ai/dsh-tool-refactory/console'
import type { ConsoleConfig as ConsoleSettings, ConsoleInvocation, ConsoleResult } from '@deepseek-ai/dsh-tool-refactory/console'
import { Remote, RemoteError, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'
import type {
  RefactoryAuthoringProgress,
  RefactoryCandidateRow,
  RefactoryGate,
  RefactoryItemResults,
  RefactoryKnob,
  RefactoryKnobKind,
  RefactoryLedgerEntry,
  RefactoryModeChoice,
  RefactoryPackageChoice,
  RefactoryPackageMode,
  RefactoryProfile,
  RefactoryProfileSpec,
  RefactoryProject,
  RefactoryRunCounters,
  RefactoryRunEvent,
  RefactoryRunStatus,
  RefactoryActivity,
  RefactoryWatchFrame,
  RefactoryRunSummary,
  RefactorySingleRun,
  RefactorySingleEvent,
  RefactorySingleRunRequest,
  RefactoryStartReceipt,
  RefactoryStartRequest,
  RefactorySteering,
  RefactorySteerRequest,
  RefactoryStopReceipt,
  RefactoryStrategies,
  RefactoryStrategyChoice,
} from './types.ts'

export type * from './types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** Host owner of the `refactoryConsole` Remote namespace. */
    refactoryConsole: RefactoryConsole
  }
}

/**
 * Deployment settings: where the checkout is and how long a call may take.
 *
 * The same four settings the model-facing tool mounts, from one schema, so a
 * deployment that points the tool at a checkout points this service there too.
 */
export type Config = ConsoleSettings

/** A JSON object, as a console answer's members decode. */
type Json = Record<string, unknown>

/**
 * Require one console answer member to be a list of JSON objects.
 *
 * The console writes JSON on stdout, which is a process boundary, so its shape
 * is checked here rather than trusted: a console whose answer changed shape must
 * fail loudly at this seam instead of rendering an empty panel.
 * @param value - the console's decoded answer.
 * @param what - the member's name, for the message.
 * @returns the member as a list of objects.
 */
function objectList(value: unknown, what: string): Json[] {
  if (!Array.isArray(value)) throw new Error(`the console's ${what} is not a list`)
  for (const item of value) {
    if (!isRecord(item)) throw new Error(`the console's ${what} holds a member that is not an object`)
  }
  return value as Json[]
}

/**
 * Require one console answer member to be a JSON object.
 * @param value - the console's decoded answer.
 * @param what - the member's name, for the message.
 * @returns the member as an object.
 */
function objectOf(value: unknown, what: string): Json {
  if (!isRecord(value)) throw new Error(`the console's ${what} is not an object`)
  return value
}

/** A string member of a console answer, or a stated absence. */
function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/** A number member of a console answer, or a stated absence. */
function count(value: unknown): number {
  return typeof value === 'number' ? value : 0
}

/** A JSON scalar as the text a knob's value or member is read and written as. */
function scalar(value: unknown): string {
  return typeof value === 'string' ? value : typeof value === 'number' || typeof value === 'boolean'
    ? String(value)
    : ''
}

/** The knob kinds `refactory_dsh.nn.knob` declares. */
const KNOB_KINDS: readonly RefactoryKnobKind[] = ['int', 'float', 'bool', 'str', 'enum']

/**
 * One knob as the catalogue declared it.
 *
 * Kinds travel as the closed set the loop declares, so a console that grew a kind
 * this picker cannot draw fails here rather than rendering a control that writes a
 * value the knob refuses.
 * @param value - one entry of a mode's `parameters`.
 * @param name - the knob's name, for the message.
 * @returns the declared knob.
 */
function knob(value: unknown, name: string): RefactoryKnob {
  const entry = objectOf(value, `knob "${name}"`)
  const kind = text(entry.kind)
  if (!KNOB_KINDS.includes(kind as RefactoryKnobKind)) {
    throw new Error(`the console's knob "${name}" declares kind "${kind}", which this picker cannot draw`)
  }
  const bound = Array.isArray(entry.range) ? entry.range : []
  const values = Array.isArray(entry.values) ? entry.values.map(scalar) : []
  const low = bound[0]
  const high = bound[1]
  return {
    value: scalar(entry.value),
    kind: kind as RefactoryKnobKind,
    values,
    range: [typeof low === 'number' ? low : null, typeof high === 'number' ? high : null],
    frozen: entry.frozen === true,
    description: text(entry.description),
  }
}

/** Every knob a catalogue entry declares, by name. */
function knobs(value: unknown, mode: string): Readonly<Record<string, RefactoryKnob>> {
  if (value === undefined) return {}
  const entries = objectOf(value, `knob declarations for "${mode}"`)
  return Object.fromEntries(Object.entries(entries).map(([name, entry]) => [name, knob(entry, name)]))
}

/** What a graph spine recorded about one run, or `null` for a run it did not drive. */
function authoring(value: unknown): RefactoryAuthoringProgress | null {
  if (!isRecord(value)) return null
  return {
    spine: text(value.spine),
    nodes_due: Array.isArray(value.nodes_due) ? value.nodes_due.map(scalar) : [],
    error: text(value.error),
  }
}

/** What a run wrote to `facts.json`, each member as text. */
function facts(value: unknown): Readonly<Record<string, string>> {
  if (!isRecord(value)) return {}
  return Object.fromEntries(
    Object.entries(value)
      .map(([name, entry]): [string, string] => [
        name,
        Array.isArray(entry) || isRecord(entry) ? JSON.stringify(entry) : scalar(entry),
      ])
      .filter(([, written]) => written !== ''),
  )
}

/** Counts only recorded item verdicts; never invents a mode's aggregate score. */
function itemResults(value: unknown): RefactoryItemResults | null {
  if (!isRecord(value)) return null
  return { count: count(value.count), passed: count(value.passed), failed: count(value.failed), unclassified: count(value.unclassified) }
}

/** Every entry of one intent ledger, as the console parsed it back. */
function ledgerEntries(value: unknown): readonly RefactoryLedgerEntry[] {
  return objectList(value, 'ledger entries').map(entry => ({
    at: text(entry.at),
    kind: text(entry.kind),
    text: text(entry.text),
  }))
}

/** A list of strings one console answer carries. */
function stringList(value: unknown, what: string): readonly string[] {
  if (!Array.isArray(value)) throw new Error(`the console's ${what} is not a list`)
  return value.map(scalar)
}

/** The `user` gates a console answer reports, each with the verdicts it accepts. */
function gateList(value: unknown): readonly RefactoryGate[] {
  return objectList(value, 'run gates').map(entry => ({
    id: text(entry.id),
    prompt: text(entry.prompt),
    verdicts: Array.isArray(entry.verdicts) ? entry.verdicts.map(scalar) : [],
  }))
}

/** The verdict recorded for each gate, by gate id. */
function gateVerdicts(value: unknown): Readonly<Record<string, string>> {
  if (!isRecord(value)) return {}
  return Object.fromEntries(Object.entries(value).map(([gate, verdict]) => [gate, scalar(verdict)]))
}

/** A durable control receipt, including refusals and recovered legacy choices. */
function controlReceipt(value: unknown): Exclude<RefactorySteering['control'], undefined> {
  if (value == null) return null
  const entry = objectOf(value, 'control receipt')
  if (entry.status !== 'queued' && entry.status !== 'applied' && entry.status !== 'refused') {
    throw new Error('the console\'s control receipt has an unknown status')
  }
  return {
    action: text(entry.action),
    status: entry.status,
    candidate: text(entry.candidate),
    target: text(entry.target),
    as: text(entry.as),
    package_root: text(entry.package_root),
    message: text(entry.message),
    error: text(entry.error),
    recovered: entry.recovered === true,
    ...isRecord(entry.recovered_control) ? {
      recovered_control: {
        action: text(entry.recovered_control.action),
        target: text(entry.recovered_control.target),
      },
    } : {},
  }
}

/** Host Remote service over the refactory console. */
export class RefactoryConsole extends TypertRemoteService {
  static inject = ['subprocess', 'typert']

  static Config: z<Config> = z.object({
    repoRoot: z.string().default(process.cwd()),
    pythonBin: z.string().default(''),
    timeoutMs: z.natural().min(1).max(120 * 60 * 1000).default(CONSOLE_DEFAULTS.timeoutMs),
    collectBytes: z.natural().min(1024).default(CONSOLE_DEFAULTS.collectBytes),
  })

  private readonly console: Console
  private readonly projectRoot: string

  /**
   * @param ctx - Host context carrying the subprocess capability.
   * @param config - where the checkout is and how long one console call may take.
   */
  constructor(ctx: Context, config: Config) {
    super(ctx, 'refactoryConsole')
    const settings = resolveConsoleSettings(config as Required<Config>)
    this.projectRoot = settings.repoRoot
    this.console = new Console(ctx.subprocess, settings)
  }

  /**
   * The configured checkout used by this console, independent of chat workspace.
   * @param signal - caller cancellation.
   * @returns the absolute project directory used for console calls.
   */
  @Remote
  async project(signal: AbortSignal): Promise<string> {
    signal.throwIfAborted()
    return this.projectRoot
  }

  /**
   * The modes a run can be started from, each with the verb that runs it.
   * @param signal - caller cancellation.
   * @returns the console's catalogue, as the choices it describes.
   */
  /** Read-only live discovery uses the same contract as the CEO and terminal. */
  @Remote
  async help(topic: string, signal: AbortSignal): Promise<Record<string, unknown>> {
    return objectOf(await this.call({ verb: 'help', options: { topic } }, signal), 'help')
  }

  /** Effective project worker defaults and provenance; never launches a worker. */
  @Remote
  async workers(signal: AbortSignal): Promise<Record<string, unknown>> {
    return objectOf(await this.call({ verb: 'workers' }, signal), 'workers')
  }

  /**
   * The registered projects, for the GUI's target picker.
   * @param signal - caller cancellation.
   * @returns the projects the registry knows.
   */
  @Remote
  async projects(signal: AbortSignal): Promise<readonly RefactoryProject[]> {
    const answer = await this.call({ verb: 'projects' }, signal)
    return objectList(answer, 'project list').map(entry => ({
      name: text(entry.name),
      path: text(entry.path),
      kind: text(entry.kind ?? 'existing'),
    }))
  }

  @Remote
  async modes(signal: AbortSignal): Promise<readonly RefactoryModeChoice[]> {
    const answer = await this.call({ verb: 'modes' }, signal)
    return objectList(answer, 'mode catalogue').map(entry => ({
      id: text(entry.id),
      package: text(entry.package),
      mode: text(entry.mode),
      description: text(entry.description),
      startable: entry.startable === true,
      enabled: entry.enabled === true,
      scored: entry.scored === true,
      action: text(entry.action),
      parameters: knobs(entry.parameters, text(entry.id)),
      ...entry.trust === undefined ? {} : { trust: text(entry.trust) },
      ...entry.reason === undefined ? {} : { reason: text(entry.reason) },
    }))
  }

  /**
   * The named outer loops, optimizers and evaluators a run's settings choose between.
   * @param signal - caller cancellation.
   * @returns the three registries, each a list of named choices.
   */
  @Remote
  async strategies(signal: AbortSignal): Promise<RefactoryStrategies> {
    const answer = objectOf(await this.call({ verb: 'strategies' }, signal), 'strategy answer')
    return {
      outer_loops: choices(answer.outer_loops, 'outer loops'),
      optimizers: choices(answer.optimizers, 'optimizers'),
      evaluators: choices(answer.evaluators, 'evaluators'),
    }
  }

  /**
   * Every run under the run root, newest first.
   * @param signal - caller cancellation.
   * @returns the runs, each with where it got to.
   */
  @Remote
  async runs(signal: AbortSignal): Promise<readonly RefactoryRunSummary[]> {
    const answer = await this.call({ verb: 'runs' }, signal)
    return objectList(answer, 'run list').map(entry => ({
      run_id: text(entry.run_id),
      root: text(entry.root),
      mode: typeof entry.mode === 'string' ? entry.mode : null,
      best_score: typeof entry.best_score === 'number' ? entry.best_score : null,
      finished: entry.finished === true,
      state: text(entry.state),
      pending_questions: count(entry.pending_questions),
    }))
  }

  /**
   * One run's candidates, ranked, with what a person did to each.
   *
   * `kept` and `rejected` travel on the row because they are what the candidate
   * controls are drawn from: a person reading the table needs to see the acts they
   * already asked for, and a live run reports them from its own `state.json`
   * because its report does not exist yet.
   * @param run - the run id or absolute run directory.
   * @param signal - caller cancellation.
   * @returns the candidates by score, highest first.
   */
  @Remote
  async leaderboard(run: string, signal: AbortSignal): Promise<readonly RefactoryCandidateRow[]> {
    const addressed = this.runAddress(run)
    const answer = await this.call({ verb: 'leaderboard', options: { run: addressed } }, signal)
    return objectList(answer, 'leaderboard').map(entry => ({
      id: text(entry.id),
      parent: typeof entry.parent === 'string' ? entry.parent : null,
      score: typeof entry.score === 'number' ? entry.score : null,
      raw_score: typeof entry.raw_score === 'number' ? entry.raw_score : null,
      nodes: count(entry.nodes),
      kept: entry.kept === true,
      rejected: entry.rejected === true,
    }))
  }

  /**
   * The packages the registry knows, with the modes each declares.
   * @param signal - caller cancellation.
   * @returns the packages as described choices.
   */
  @Remote
  async packages(signal: AbortSignal): Promise<readonly RefactoryPackageChoice[]> {
    return this.catalogue(signal)
  }

  /**
   * Start a run from one described mode.
   *
   * The mode must be one the catalogue offers and one whose `action` is `start`;
   * an authoring mode is refused by name, with the verb that runs it, because a
   * person who picked it needs the action rather than a failed process.
   * @param request - the mode, run addressing, settings, and starting knobs.
   * @param signal - caller cancellation.
   * @returns what the console recorded about the start.
   */
  @Remote
  async start(request: RefactoryStartRequest, signal: AbortSignal): Promise<RefactoryStartReceipt> {
    const mode = await this.runnable(request.mode, signal)
    const invocation = startInvocation({
      mode,
      runId: request.runId,
      runsRoot: request.runsRoot,
      settings: request.settings,
      knobs: request.knobs,
    })
    return receipt(await this.call(invocation, signal))
  }

  /** List saved recipes, including older immutable versions. */
  @Remote
  async profiles(signal: AbortSignal): Promise<readonly RefactoryProfile[]> {
    return await this.call({ verb: 'profiles' }, signal) as readonly RefactoryProfile[]
  }

  /** Save explicit edits to settings or objective as a new version. */
  @Remote
  async profileSave(name: string, spec: RefactoryProfileSpec, signal: AbortSignal): Promise<RefactoryProfile> {
    return await this.call({ verb: 'profile-save', options: { name, spec: JSON.stringify(spec) } }, signal) as RefactoryProfile
  }

  /** Preview uses the same resolver as start. */
  @Remote
  async profileResolve(request: RefactoryStartRequest, signal: AbortSignal): Promise<Record<string, unknown>> {
    const invocation = startInvocation(request)
    return objectOf(await this.call({ ...invocation, verb: 'profile-resolve' }, signal), 'resolved profile')
  }

  /** Inspect failures and missing measurements alongside successful candidates. */
  @Remote
  async comparison(run: string, signal: AbortSignal): Promise<Record<string, unknown>> {
    return objectOf(await this.call({ verb: 'comparison', options: { run: this.runAddress(run) } }, signal), 'candidate comparison')
  }

  /**
   * Use one described mode once, with a person's knobs, and report what it left.
   *
   * This is the console's `run` — a person using a mode — where `start` is the
   * outer loop searching one. Any enabled, described mode can run once, including
   * authoring modes that do not emit a score and cannot be searched.
   *
   * The run is synchronous: the console drives it to quiescence before answering.
   * A run that reached one of the spine's named terminal stops — `failed`,
   * `refused` or `halted` — is a result: the answer carries that word, so it is
   * reported rather than raised. A `user` gate is one of those and not an exception:
   * this call is always headless, so a gate a person owns that nobody answered is
   * recorded `refused`, never left open. A run that stopped with work still due is
   * not a terminal stop — the graph did not finish — and is raised.
   * @param request - the mode and the knob values it uses.
   * @param signal - caller cancellation.
   * @returns where the run got to, and what it recorded.
   */
  @Remote
  async run(request: RefactorySingleRunRequest, signal: AbortSignal): Promise<RefactorySingleRun> {
    const mode = await this.runnable(request.mode, signal, false)
    return this.executeSingle(mode, request.knobs, signal, undefined, request.settings)
  }

  /**
   * Stream one unique run's identity while the same console call owns execution.
   * A prepared event certifies only a fresh lowering, never model authentication
   * or success. Ending this stream aborts and awaits its owned console call.
   * @param request - admitted mode and knob values; callers cannot choose a path.
   * @param signal - the transport's cancellation for this stream.
   * @returns starting identity, durable preparation, and the ordinary final answer.
   */
  @Remote({ mode: 'stream' })
  async *runStream(request: RefactorySingleRunRequest, signal: AbortSignal): AsyncGenerator<RefactorySingleEvent> {
    const mode = await this.runnable(request.mode, signal, false)
    signal.throwIfAborted()
    const project = await realpath(this.projectRoot)
    const factory = join(project, '.factory')
    await mkdir(factory, { recursive: true })
    if (await realpath(factory) !== factory) {
      throw new RemoteError('gateway/bad-request', 'the factory directory must remain inside the configured project', {})
    }
    const parent = join(factory, 'interactive')
    await mkdir(parent, { recursive: true })
    if (await realpath(parent) !== parent) {
      throw new RemoteError('gateway/bad-request', 'the one-shot run directory must remain inside the configured project', {})
    }
    const root = join(parent, `single-${randomUUID()}`)
    // Exclusive mkdir claims a fresh namespace: no caller path or reused result.
    await mkdir(root)
    const controller = new AbortController()
    const ownedSignal = AbortSignal.any([signal, controller.signal])
    let settled = false
    const completion = this.executeSingle(mode, request.knobs, ownedSignal, root, request.settings).then(
      run => ({ ok: true as const, run }),
      error => ({ ok: false as const, error }),
    ).finally(() => { settled = true })
    try {
      yield { kind: 'launch', launch: { mode, root, phase: 'starting' } }
      while (!settled) {
        ownedSignal.throwIfAborted()
        let invocation: unknown
        try {
          invocation = JSON.parse(await readFile(join(root, '.refactory', 'invocation.json'), 'utf8'))
        } catch (error) {
          // An incomplete external write is not preparation evidence or a reason
          // to cancel otherwise valid work. Keep the same bounded observation.
          if (!(error instanceof SyntaxError)
            && !(error instanceof Error && 'code' in error && error.code === 'ENOENT')) throw error
        }
        // prepare() publishes this record last, after its graph/declarations.
        if (isRecord(invocation) && typeof invocation.id === 'string'
          && invocation.mode === mode.replace('/', '-') && invocation.previous === null) {
          yield { kind: 'launch', launch: { mode, root, phase: 'prepared', invocation: invocation.id } }
          break
        }
        // Stop observation at settlement/abort; the timer is bounded and owned.
        const tick = new AbortController()
        try {
          await Promise.race([completion, delay(100, undefined, { signal: AbortSignal.any([ownedSignal, tick.signal]) })])
        } finally { tick.abort() }
      }
      const result = await completion
      ownedSignal.throwIfAborted()
      if (!result.ok) throw result.error
      yield { kind: 'result', run: result.run }
    } finally {
      controller.abort()
      await completion
    }
  }

  /** Execute and interpret one console call for unary and streaming callers. */
  private async executeSingle(mode: string, knobs: readonly string[] | undefined, signal: AbortSignal, workdir?: string, settings?: Readonly<Record<string, string | number>>): Promise<RefactorySingleRun> {
    const result = await this.consoleCall(runInvocation({ mode, knobs, workdir, settings }), signal)
    const answer = isRecord(result.value) ? result.value : {}
    const single = answer.run
    const progress = isRecord(single) ? authoring(single.authoring) : null
    const failed = !result.ok || answer.ok === false
      || (typeof answer.exit_code === 'number' && answer.exit_code !== 0)
    const failure = !result.ok ? result.error
      : typeof answer.error === 'string' ? answer.error : `the run of "${mode}" failed (exit ${String(answer.exit_code)})`
    const stopped = progress !== null && ['failed', 'refused', 'halted'].includes(progress.spine)
    if (!isRecord(single) || (failed && !stopped)) {
      // Preparing a directory is not a successful launch. In particular, missing
      // or previously successful spine status cannot mask a new process failure.
      throw new RemoteError(
        'refactory/console-failed',
        failed ? failure : `the console left no run directory for "${mode}"`,
        { argv: result.argv, ...(isRecord(single) ? { run: single.root } : {}) },
      )
    }
    return {
      run_id: text(single.run_id),
      root: text(single.root),
      mode: typeof single.mode === 'string' ? single.mode : null,
      state: failed ? 'failed' : text(single.state),
      finished: failed || single.finished === true,
      best_score: typeof single.best_score === 'number' ? single.best_score : null,
      authoring: failed && progress !== null ? { ...progress, error: failure } : progress,
      // New consoles render before the JSON bridge so exact integers, including
      // nested values, survive JavaScript's number range. Keep older consoles usable.
      facts: facts(isRecord(single.facts_text) ? single.facts_text : single.facts),
      item_results: itemResults(single.item_results),
    }
  }

  /**
   * Where one run is, and what is new since the reader last looked.
   *
   * The two cursors index different files and never derive from one another
   * (R-4.3), so they travel as two flags: `--after` for the progress delta and
   * `--activity-after` for the journal page. `activityAfter` defaults to `after`
   * so a first reader still gets a journal page seeded from where it stopped,
   * but a streaming caller passes its own journal cursor — feeding the journal
   * index into `--after` would drive the progress delta off a count that can
   * exceed the progress file and empty `events` for the whole stream. The page
   * size is this service's own choice, passed explicitly, so the console's clamp
   * stays a safety net rather than the policy owner.
   * @param run - the run id or absolute run directory.
   * @param after - the progress cursor the caller has already read.
   * @param signal - caller cancellation.
   * @param activityAfter - the journal cursor to page from; defaults to `after`.
   * @returns the run's state, the new events, and the offset for the next look.
   */
  @Remote
  async status(run: string, after: number, signal: AbortSignal, activityAfter: number = after): Promise<RefactoryRunStatus> {
    const addressed = this.runAddress(run)
    const answer = objectOf(
      await this.call({
        verb: 'status',
        options: { run: addressed, after, 'activity-after': activityAfter, 'activity-limit': ACTIVITY_PAGE },
      }, signal),
      'run status',
    )
    return {
      run_id: text(answer.run_id),
      root: text(answer.root),
      mode: typeof answer.mode === 'string' ? answer.mode : null,
      kind: answer.kind === 'optimizer' || answer.kind === 'graph' ? answer.kind : 'unknown',
      state: text(answer.state),
      events: events(answer.events),
      cursor: count(answer.cursor),
      counters: counters(answer.counters),
      best_score: typeof answer.best_score === 'number' ? answer.best_score : null,
      finished: answer.finished === true,
      running: answer.running === true,
      authoring: authoring(answer.authoring),
      item_results: itemResults(answer.item_results),
      activity: isRecord(answer.activity) ? answer.activity as unknown as RefactoryActivity : null,
    }
  }

  /** Preview invalidation or queue a control for the owning graph boundary. */
  @Remote
  async graphControl(request: { run: string; action: string; node?: string; text?: string; revision?: string }, signal: AbortSignal): Promise<Record<string, unknown>> {
    return objectOf(await this.call({ verb: 'graph-control', positional: [request.action], options: {
      run: this.runAddress(request.run), node: request.node ?? '', text: request.text ?? '', revision: request.revision ?? '',
    } }, signal), 'graph control receipt')
  }

  /**
   * Stream persisted changes; closing this reader never cancels its run.
   *
   * The two cursors are the stream's own state, not the caller's: the caller
   * hands one `after` to start from, and every later poll advances each cursor
   * from the answer that reported it — the progress delta from `status.cursor`,
   * the journal page from `status.activity.cursor`. They index different files
   * and never derive from one another (R-4.3), so advancing one from the other
   * would re-read the journal from the same index on every poll — the whole cost
   * this paging exists to remove — or drive the progress delta off a journal
   * count the progress file need not reach. A reset to 0 mid-stream would splice
   * the same records onto the history twice.
   * @param run - the run id or absolute run directory.
   * @param after - the cursor the caller's history already reaches.
   * @param signal - the transport's cancellation for this stream.
   * @returns each changed frame of the run, and the steering read beside it.
   */
  @Remote({ mode: 'stream' })
  async *watchStream(run: string, after: number, signal: AbortSignal): AsyncGenerator<RefactoryWatchFrame> {
    let previous = ''
    let progressAfter = after
    let activityAfter = after
    while (!signal.aborted) {
      const status = await this.status(run, progressAfter, signal, activityAfter)
      const steering = await this.steer({ run }, signal)
      // Advance before yielding, so a frame's own cursors are what this poll asked
      // for rather than values the next poll has already moved past.
      progressAfter = status.cursor
      activityAfter = status.activity?.cursor ?? activityAfter
      const frame = { status, steering }
      const encoded = JSON.stringify(frame)
      if (encoded !== previous) { yield frame; previous = encoded }
      // A finished graph may be explicitly reopened while this view is attached.
      // Keep its reader alive until the tab detaches; idle terminal runs poll less often.
      const terminal = status.finished || ['failed', 'died', 'halted'].includes(status.state)
      try { await delay(terminal ? 3000 : 1000, undefined, { signal }) } catch { return }
    }
  }

  /**
   * Ask one run to stop.
   * @param run - the run id or absolute run directory.
   * @param signal - caller cancellation.
   * @returns whether the run's process is gone.
   */
  @Remote
  async stop(run: string, signal: AbortSignal): Promise<RefactoryStopReceipt> {
    const addressed = this.runAddress(run)
    const answer = objectOf(
      await this.call({ verb: 'stop', options: { run: addressed } }, signal),
      'stop answer',
    )
    return { stopped: answer.stopped === true }
  }

  /**
   * Read one run's intent ledger, and optionally append to it.
   *
   * This is the console's `steer`: a note, a correction, a constraint, or the verdict
   * that answers a gate a run is stopped at. The gates travel back on the same call
   * because the console owns both formats — the ledger's and the run's lowered graph
   * — so the panel can offer the answer a `user` gate accepts without reading either
   * one itself. A run nobody steered leaves the ledger absent, which is reported as
   * an absence rather than an error.
   * @param request - the run whose ledger is read, and what to append to it.
   * @param signal - caller cancellation.
   * @returns the ledger's entries, the recorded verdicts, and the gates now waiting.
   */
  @Remote
  async steer(request: RefactorySteerRequest, signal: AbortSignal): Promise<RefactorySteering> {
    const addressed = this.runAddress(request.run)
    const note = (request.note ?? '').trim()
    const verdicts = (request.verdicts ?? []).map(entry => entry.trim()).filter(entry => entry !== '')
    const invocation: ConsoleInvocation = {
      verb: 'steer',
      options: {
        run: addressed,
        // An empty note is not a note: the console refuses one, and a caller that
        // asked to read the ledger must not accidentally try to write to it.
        ...note === '' ? {} : { note },
        ...request.kind === undefined ? {} : { kind: request.kind },
        ...verdicts.length === 0 ? {} : { verdict: verdicts },
        // Control is its own flag, not a note: the loop polls it, and a pause must
        // not have to be read as prose. The three population acts travel the same
        // way, carrying the candidate they name and the package an adopt writes.
        ...this.controlOption(request.control, request.target),
        ...request.lineage === true ? { lineage: true } : {},
        ...(request.as ?? '').trim() === '' ? {} : { as: (request.as ?? '').trim() },
        // Read the waiting gates on the same call, so appending a note leaves the
        // section that offers a verdict current rather than a frame behind.
        gates: true,
      },
    }
    const answer = objectOf(await this.call(invocation, signal), 'steering answer')
    return {
      run: text(answer.run),
      ledger: text(answer.ledger),
      exists: answer.exists === true,
      entries: ledgerEntries(answer.entries),
      constraints: stringList(answer.constraints, 'ledger constraints'),
      verdicts: gateVerdicts(answer.verdicts),
      gates: gateList(answer.gates),
      problems: stringList(answer.problems, 'ledger problems'),
      finished: answer.finished === true,
      control: controlReceipt(answer.control),
    }
  }

  /**
   * One control act as the `steer` verb writes it: a flag, and the candidate it
   * names when it names one.
   *
   * The act is refused here rather than written for the loop to guess at, because a
   * request that cannot be honored is a person's mistake to fix and the file has no
   * way to say "that id was wrong" other than a run that quietly did nothing.
   */
  private controlOption(
    control: RefactorySteerRequest['control'],
    target: string | undefined,
  ): Readonly<Record<string, string | boolean>> {
    if (control === undefined) return {}
    if (control !== 'keep' && control !== 'reject') return { [control]: true }
    const named = (target ?? '').trim()
    if (named === '') {
      throw new RemoteError('gateway/bad-request', `${control} needs the candidate id it names`, {})
    }
    return { [control]: named }
  }

  /**
   * Enable or disable one package, and answer with the catalogue as it now stands.
   * @param name - the package name, as the catalogue lists it.
   * @param enabled - the state to set.
   * @param signal - caller cancellation.
   * @returns the packages as described choices, after the change.
   */
  @Remote
  async setPackageEnabled(
    name: string,
    enabled: boolean,
    signal: AbortSignal,
  ): Promise<readonly RefactoryPackageChoice[]> {
    const target = this.packageName(name)
    await this.registry(enabled ? 'enable' : 'disable', target, '', signal)
    return this.catalogue(signal)
  }

  /**
   * Remove one workspace package, and answer with the catalogue as it now stands.
   * @param name - the package name, as the catalogue lists it.
   * @param signal - caller cancellation.
   * @returns the packages as described choices, after the change.
   */
  @Remote
  async removePackage(name: string, signal: AbortSignal): Promise<readonly RefactoryPackageChoice[]> {
    const target = this.packageName(name)
    await this.registry('remove', target, '--force', signal)
    return this.catalogue(signal)
  }

  /**
   * Bring a package in from a directory or git URL, and answer with the catalogue
   * as it now stands.
   * @param source - the directory or git URL to import from.
   * @param name - the name to import it as; empty keeps the package's own name.
   * @param signal - caller cancellation.
   * @returns the packages as described choices, after the import.
   */
  @Remote
  async importPackage(
    source: string,
    name: string,
    signal: AbortSignal,
  ): Promise<readonly RefactoryPackageChoice[]> {
    const from = source.trim()
    if (from.length === 0) {
      throw new RemoteError('gateway/bad-request', 'source is required: a directory or git URL to import from', {})
    }
    const as = name.trim()
    await this.registry('import', from, as.length === 0 ? '' : `--as ${JSON.stringify(as)}`, signal)
    return this.catalogue(signal)
  }

  /**
   * Run one registry verb through the console.
   * @param action - the registry verb.
   * @param target - the package or source it acts on.
   * @param args - extra registry flags, as a shell-quoted string.
   * @param signal - caller cancellation.
   * @returns the forwarded entry point's answer.
   */
  private async registry(action: string, target: string, args: string, signal: AbortSignal): Promise<Json> {
    const invocation = registryInvocation(action, target, args)
    return objectOf(await this.call(invocation, signal), 'registry answer')
  }

  /**
   * The registry's catalogue, as described choices.
   *
   * The registry lists modes grouped by package and reports the package's
   * enabled state and location on each mode, so a package that declares no modes
   * reports no state: its `enabled` is `null` rather than a guess.
   * @param signal - caller cancellation.
   * @returns the packages, each with the modes it declares.
   */
  private async catalogue(signal: AbortSignal): Promise<readonly RefactoryPackageChoice[]> {
    const answer = await this.registry('list', '', '', signal)
    if (answer.ok !== true) {
      const failure = objectOf(answer.result, 'registry failure')
      throw new RemoteError(
        'refactory/console-failed',
        text(failure.error) || text(failure.output) || 'the registry could not list packages',
        {},
      )
    }
    const listed = objectOf(answer.result, 'registry catalogue')
    return Object.entries(listed)
      .map(([name, modes]): RefactoryPackageChoice => {
        const declared = objectList(modes, `catalogue for "${name}"`)
        const first = declared[0]
        return {
          name,
          enabled: first === undefined ? null : first.package_enabled === true,
          where: text(first?.where),
          modes: declared.map((entry): RefactoryPackageMode => ({
            name: text(entry.name),
            description: text(entry.description),
            trust: text(entry.trust),
          })),
        }
      })
      .sort((left, right) => left.name.localeCompare(right.name))
  }

  /**
   * The mode one run-scoped verb may act on, or the refusal that says what to do.
   *
   * The catalogue's own `action` field decides which verb runs a mode, so an
   * authoring mode is refused with the verb that runs it rather than started and
   * reported as a failed process.
   * @param requested - the mode the caller asked for.
   * @param signal - caller cancellation.
   * @returns the trimmed mode identifier to pass to the console.
   */
  private async runnable(requested: string, signal: AbortSignal, search = true): Promise<string> {
    const mode = requested.trim()
    if (mode.length === 0) {
      throw new RemoteError('gateway/bad-request', 'mode is required: the <package>/<name> to run', {})
    }
    const choices = await this.modes(signal)
    const choice = choices.find(entry => entry.id === mode)
    if (choice === undefined) {
      throw new RemoteError(
        'refactory/unknown-mode',
        `"${mode}" is not in the mode catalogue; ask for the modes and choose one of them`,
        { mode },
      )
    }
    if (!choice.enabled) {
      throw new RemoteError(
        'refactory/package-disabled',
        `"${choice.package}" is disabled, so "${mode}" cannot run; enable the package first`,
        { mode, package: choice.package },
      )
    }
    if (search && choice.action !== 'start') {
      throw new RemoteError(
        'refactory/authoring-mode',
        `"${mode}" is an authoring mode, not a mode to run or search; use the "${choice.action}" action`,
        { mode, action: choice.action },
      )
    }
    return mode
  }

  /**
   * One console call, with its refusal turned into a Remote error.
   *
   * A truncated payload is its own code, not the console's refusal: nothing was
   * refused — the collection cap cut stdout before an answer printed — and a
   * caller that renders it as `console-failed` would show "printed no JSON" for
   * a size problem it can fix. The cap and the retained tail travel with it so
   * the notice can name both.
   * @param invocation - the verb, positionals, and options.
   * @param signal - caller cancellation.
   * @returns the console's decoded JSON.
   */
  private async call(invocation: ConsoleInvocation, signal: AbortSignal): Promise<unknown> {
    const result = await this.consoleCall(invocation, signal)
    if (!result.ok) {
      if (result.truncation !== undefined) {
        throw new RemoteError('refactory/console-truncated', result.error, {
          argv: result.argv,
          capBytes: result.truncation.capBytes,
          retainedBytes: result.truncation.retainedBytes,
        })
      }
      throw new RemoteError('refactory/console-failed', result.error, { argv: result.argv })
    }
    return result.value
  }

  /**
   * One console call that keeps its answer even when the console failed.
   *
   * `run` reports where a run stopped in the same payload as the reason it did not
   * finish, so a caller reading that payload cannot take the error-shaped path.
   * An answer that only arrived by dropping the paging options says so on its
   * `activity`, because an unpaged journal is something the reader has to be
   * told about rather than something it can infer from a missing member.
   * @param invocation - the verb, positionals, and options.
   * @param signal - caller cancellation.
   * @returns the console's answer, successful or not.
   */
  private async consoleCall(invocation: ConsoleInvocation, signal: AbortSignal): Promise<ConsoleResult> {
    if (signal.aborted) throw new RemoteError('gateway/cancelled', 'the refactory call was cancelled', {})
    const result = await this.console.run(invocation, signal)
    const suppressed = result.suppressed ?? []
    if (suppressed.length === 0 || !result.ok || !isRecord(result.value) || !isRecord(result.value.activity)) {
      return result
    }
    return { ...result, value: { ...result.value, activity: { ...result.value.activity, paged: false } } }
  }

  /**
   * The run address one run-scoped verb needs.
   * @param run - the run id or absolute run directory.
   * @returns the address to pass to the console.
   */
  private runAddress(run: string): string {
    const addressed = run.trim()
    if (addressed.length === 0) {
      throw new RemoteError('gateway/bad-request', 'run is required: the run id or its directory', {})
    }
    return addressed
  }

  /**
   * The package name one registry verb needs.
   * @param name - the package name, as the catalogue lists it.
   * @returns the name to pass to the console.
   */
  private packageName(name: string): string {
    const named = name.trim()
    if (named.length === 0) {
      throw new RemoteError('gateway/bad-request', 'name is required: the package to act on', {})
    }
    return named
  }
}

/**
 * Journal records one `status` answer carries, the same window the panel renders.
 *
 * Sent explicitly rather than left to the console's default: the service decides
 * how much history one poll frame holds, and the console's clamp stays a bound on
 * a caller's mistake instead of a policy this service inherits by accident.
 */
const ACTIVITY_PAGE = 200

/** The strategy choices one level-two registry answered with. */
function choices(value: unknown, what: string): readonly RefactoryStrategyChoice[] {
  return objectList(value, what).map(entry => ({
    name: text(entry.name),
    description: text(entry.description),
  }))
}

/** The progress events of one status answer. */
function events(value: unknown): readonly RefactoryRunEvent[] {
  return objectList(value, 'run events').map(entry => ({
    ...typeof entry.epoch === 'number' ? { epoch: entry.epoch } : {},
    ...typeof entry.generation === 'number' ? { generation: entry.generation } : {},
    ...typeof entry.evaluations === 'number' ? { evaluations: entry.evaluations } : {},
    ...typeof entry.score === 'number' ? { score: entry.score } : {},
  }))
}

/** The counters of one status answer. */
function counters(value: unknown): RefactoryRunCounters {
  const entry = objectOf(value, 'run counters')
  return {
    epoch: count(entry.epoch),
    generation: count(entry.generation),
    evaluations: count(entry.evaluations),
    steps: count(entry.steps),
  }
}

/** What the console recorded about a start. */
function receipt(value: unknown): RefactoryStartReceipt {
  const entry = objectOf(value, 'start answer')
  const config = isRecord(entry.config) ? entry.config : {}
  return {
    run_id: text(entry.run_id),
    mode: text(entry.mode),
    root: text(entry.root),
    started: count(entry.started),
    config: Object.fromEntries(Object.entries(config).map(([key, setting]) => [key, text(setting)])),
    log: text(entry.log),
    pid: count(entry.pid),
  }
}

export default RefactoryConsole
