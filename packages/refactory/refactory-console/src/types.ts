/**
 * What a picker chooses between, as `refactory_dsh.console` describes it.
 *
 * Every field here is the console's own: the catalogue's `action` field decides
 * which verb runs a mode, `enabled` is the registry's answer, and the strategy
 * names are the registries'. Nothing in this file re-derives a fact the console
 * already reports, so a picker that renders these choices and a model that reads
 * the same JSON cannot disagree about what is on offer.
 *
 * The console's answer carries more than the picker reads — a mode's nodes and
 * ports stay on the wire untouched — so these are the reported choices, not a
 * second answer to the console's question.
 * @module @deepseek-ai/dsh-refactory-console/types
 */

import type { RemoteResult } from '@deepseek-ai/dsh-typert-protocol'

/** The knob kinds `refactory_dsh.nn.knob` declares. */
export type RefactoryKnobKind = 'int' | 'float' | 'bool' | 'str' | 'enum'

/** One declared knob of a mode, as the console's catalogue describes it. */
export interface RefactoryKnob {
  /** The value the mode runs with until one is chosen for it, as text. */
  readonly value: string
  /** The value kind the knob accepts. */
  readonly kind: RefactoryKnobKind
  /**
   * The finite set the knob accepts, as text: an `enum`'s members, or the values
   * a `str` knob declares. Empty means the set is open (a `str` knob) or bound by
   * `range` instead (a number).
   */
  readonly values: readonly string[]
  /** The numeric bounds `[low, high]`, either of which may be absent. */
  readonly range: readonly [number | null, number | null]
  /**
   * Whether the search may not move it. A person running the mode may still say
   * what it is, which is how an input-taking mode is used at all.
   */
  readonly frozen: boolean
  /** What the knob is for, from its own declaration. */
  readonly description: string
}

/** One mode the console's catalogue offers. */
export interface RefactoryModeChoice {
  /** `<package>/<mode>`, the identifier every console verb accepts. */
  readonly id: string
  /** Package the mode is declared in. */
  readonly package: string
  /** The mode's own name inside its package. */
  readonly mode: string
  /** The mode's one-line description. */
  readonly description: string
  /** Whether a run can be started from it right now. */
  readonly startable: boolean
  /** Whether its package is enabled. */
  readonly enabled: boolean
  /** Whether the mode declares a score, so the loop has something to search. */
  readonly scored: boolean
  /** The verb that runs this mode: `start`, or the authoring verb `create`, `edit` or `design`. */
  readonly action: string
  /** Every knob the mode declares, by name, as its own declaration describes it. */
  readonly parameters: Readonly<Record<string, RefactoryKnob>>
  /** The package's trust level, when the catalogue resolved the mode. */
  readonly trust?: string
  /** `passed`, or `pending` while a package a create or edit run registered has not passed its smoke. */
  readonly smoke?: string
  /** Why the mode could not be resolved, for a mode whose package fails to import. */
  readonly reason?: string
}

/** One named choice in a level-two registry. */
export interface RefactoryStrategyChoice {
  /** The name a run setting accepts. */
  readonly name: string
  /** What the implementation says it is, from its own first docstring line. */
  readonly description: string
}

/** The three level-two seams, each a list of described choices. */
export interface RefactoryStrategies {
  /** The named outer loops. */
  readonly outer_loops: readonly RefactoryStrategyChoice[]
  /** The named optimizers. */
  readonly optimizers: readonly RefactoryStrategyChoice[]
  /** The named evaluators. */
  readonly evaluators: readonly RefactoryStrategyChoice[]
}

/** One run under the run root, as the console lists it. */
export interface RefactoryRunSummary {
  readonly state?: string
  readonly pending_questions?: number
  /** The run directory's name. */
  readonly run_id: string
  /** Absolute path of the run directory. */
  readonly root: string
  /** The mode the run was started from; absent for a run with no recorded mode. */
  readonly mode?: string | null
  /** Best score so far, or absent when nothing scored yet. */
  readonly best_score?: number | null
  /** Whether the run wrote its report. */
  readonly finished: boolean
}

/** One progress line the loop recorded, as much of it as a reader reads. */
export interface RefactoryRunEvent {
  /** The epoch the step belongs to. */
  readonly epoch?: number
  /** The generation within the epoch. */
  readonly generation?: number
  /** Evaluations completed when the step was written. */
  readonly evaluations?: number
  /** The score the step recorded. */
  readonly score?: number
}

/**
 * One candidate on a run's leaderboard, as much of it as a reader reads.
 *
 * The console reports more per row — the full diff against the parent, the edits,
 * the size cost — and a reader that wants the diff asks for the candidate by id.
 * `kept` and `rejected` are the person's own two lists, which is what the
 * candidate controls are drawn from.
 */
export interface RefactoryCandidateRow {
  /** The candidate id every steer act and every read accepts. */
  readonly id: string
  /** The candidate this one was proposed against; `null` for the seed. */
  readonly parent: string | null
  /** The score the search compared, size cost taken off it; `null` before it is run. */
  readonly score: number | null
  /** The unpenalised score the evaluator reported; `null` before it is run. */
  readonly raw_score: number | null
  /** How many nodes the candidate's graph holds, which is what the size cost counts. */
  readonly nodes: number
  /** Whether a person pinned this candidate as a parent. */
  readonly kept: boolean
  /** Whether a person dismissed this candidate, so nothing selects it again. */
  readonly rejected: boolean
}

/** The tail counters a run's progress reports. */
export interface RefactoryRunCounters {
  /** Latest epoch in the events read. */
  readonly epoch: number
  /** Latest generation in the events read. */
  readonly generation: number
  /** Latest evaluation count in the events read. */
  readonly evaluations: number
  /** How many events were read in this frame. */
  readonly steps: number
}

/** Where one run is, and what is new since the reader last looked. */
export interface RefactoryRunStatus {
  /** Durable activity; absent on runs from older releases. */
  readonly activity?: RefactoryActivity | null
  /** Persisted run layout; unrecognized directories expose no control actions. */
  readonly kind: 'optimizer' | 'graph' | 'unknown'
  /** The run directory's name. */
  readonly run_id: string
  /** Absolute path of the run directory. */
  readonly root: string
  /** The mode the run was started from. */
  readonly mode?: string | null
  /** One of `starting`, `running`, `finished`, `died`. */
  readonly state: string
  /** Progress lines after the requested offset. */
  readonly events: readonly RefactoryRunEvent[]
  /** Offset to pass as `after` on the next look. */
  readonly cursor: number
  /** Counters as of the last event in this frame. */
  readonly counters: RefactoryRunCounters
  /** Best score so far, or `null` when nothing scored yet. */
  readonly best_score: number | null
  /** Whether the run wrote its report. */
  readonly finished: boolean
  /** Whether the run's process is alive. */
  readonly running: boolean
  /** Graph progress for a chat-driven mode; null for an optimizer search. */
  readonly authoring?: RefactoryAuthoringProgress | null
  /** Recorded Task verdicts, when this run has per-item results. */
  readonly item_results?: RefactoryItemResults | null
}

/** Read-only topology from the watched run's persisted spine checkpoint. */
export interface RefactoryRunGraph {
  readonly skipped?: readonly string[]
  readonly nodes: readonly { readonly id: string; readonly kind: string }[]
  readonly edges: readonly { readonly source: string; readonly target: string; readonly label: string }[]
}

export interface RefactoryActivity {
  readonly schema_version: number
  readonly run_id: string
  readonly invocation: string | null
  /** Index of the next journal record: where the next page starts. */
  readonly cursor: number
  /**
   * Index this answer's `events` starts at.
   *
   * Absent on a console that does not page the journal: absent means "this
   * console serves the history in one answer", which is not the same statement
   * as zero.
   */
  readonly after?: number
  /** Complete records in the journal; absent on a console that does not page. */
  readonly total?: number
  /** Records one answer may carry; absent on a console that does not page. */
  readonly page_size?: number
  /** Whether records remain past `cursor`; absent on a console that does not page. */
  readonly has_more?: boolean
  /**
   * Whether the history is paged at all.
   *
   * `false` when the read only succeeded by dropping the paging options an older
   * console refuses, so a reader shows the whole `events` it received and says
   * the history is unpaged rather than treating the answer as a complete journal.
   */
  readonly paged?: boolean
  readonly last_event_at: string | null
  readonly started_at: string | null
  readonly nodes: readonly string[]
  readonly completed: readonly string[]
  /** Absent on older hosts and optimizer roots without a graph checkpoint. */
  readonly graph?: RefactoryRunGraph | null
  readonly wait_reason: string | null
  readonly workers: readonly Readonly<Record<string, unknown>>[]
  readonly deliveries: readonly Readonly<Record<string, unknown>>[]
  readonly events: readonly Readonly<Record<string, unknown>>[]
  readonly outputs: Readonly<Record<string, unknown>>
  readonly resources: Readonly<Record<string, unknown>> | null
  readonly control: Readonly<Record<string, unknown>> | null
}

export interface RefactoryWatchFrame {
  readonly status: RefactoryRunStatus
  readonly steering: RefactorySteering
}

/** What the console recorded when it started a run. */
export interface RefactoryStartReceipt {
  /** The run directory's name. */
  readonly run_id: string
  /** The mode the run was started from. */
  readonly mode: string
  /** Absolute path of the run directory. */
  readonly root: string
  /** Unix time the start was recorded. */
  readonly started: number
  /** The run settings, as the console recorded them. */
  readonly config: Readonly<Record<string, string>>
  /** The run's log file name inside its directory. */
  readonly log: string
  /** The run process's id. */
  readonly pid: number
}

/** What asking a run to stop produced. */
export interface RefactoryStopReceipt {
  /** Whether the run's process is gone. */
  readonly stopped: boolean
}

/** The kinds a note can be, as the console's ledger declares them. */
export type RefactoryNoteKind = 'info' | 'correction' | 'constraint'

/** One entry of a run's intent ledger, as the ledger's owner parses it back. */
export interface RefactoryLedgerEntry {
  /** When the entry was recorded, as the ledger wrote it. */
  readonly at: string
  /** The entry's kind: a note kind, `Initial Idea`, or `verdict`. */
  readonly kind: string
  /** What the entry says. */
  readonly text: string
}

/**
 * One `user` gate a run is stopped at, with the verdict labels it accepts.
 *
 * A gate a person owns is the one place a run waits on a reader, so the labels
 * travel with it: they are the gate's own outgoing outcomes, and a verdict outside
 * them would be refused by the runtime.
 */
export interface RefactoryGate {
  /** The gate node's id. */
  readonly id: string
  /** What the gate asks, from the mode's own declaration; empty when it says nothing. */
  readonly prompt: string
  /** The verdict labels the gate routes on, in the graph's edge order. */
  readonly verdicts: readonly string[]
}

/** A run's intent ledger, and — when asked — the gates it is stopped at. */
export interface RefactorySteering {
  /** Absolute path of the run directory the ledger belongs to. */
  readonly run: string
  /** Absolute path of the ledger file. */
  readonly ledger: string
  /** Whether the ledger exists and has anything in it. */
  readonly exists: boolean
  /** Every entry, oldest first. */
  readonly entries: readonly RefactoryLedgerEntry[]
  /** The text of every entry whose kind is `constraint`. */
  readonly constraints: readonly string[]
  /** The last verdict recorded for each gate, by gate id. */
  readonly verdicts: Readonly<Record<string, string>>
  /** The `user` gates the run is stopped at now, and what each accepts. */
  readonly gates: readonly RefactoryGate[]
  /** What is wrong with the ledger, as the console's own fidelity check reports it. */
  readonly problems: readonly string[]
  /** A completed optimizer has no future boundary to consume live controls. */
  readonly finished?: boolean
  /** Pending or most recent control, retained across panel refreshes. */
  readonly control?: {
    readonly action: string
    readonly status: 'queued' | 'applied' | 'refused'
    readonly candidate?: string
    readonly target?: string
    readonly as?: string
    readonly package_root?: string
    readonly message?: string
    readonly error?: string
    readonly recovered?: boolean
    readonly recovered_control?: { readonly action: string; readonly target?: string }
  } | null
}

/** One read of, or addition to, a run's intent ledger. */
export interface RefactorySteerRequest {
  /** The run id or absolute run directory. */
  readonly run: string
  /** What the person said; absent or empty reads the ledger without writing. */
  readonly note?: string
  /** What the note is; the console's own default when absent. */
  readonly kind?: RefactoryNoteKind
  /**
   * Gate verdicts to record, as `GATE=VALUE`; one `--verdict` per element. Absent or
   * empty writes none, so the same call reads a ledger it did not touch.
   */
  readonly verdicts?: readonly string[]
  /**
   * Control the run itself rather than its ledger: `pause` holds it at its next
   * generation boundary, `resume` releases it, `stop` ends it with the reason
   * recorded, `keep` pins a candidate as a parent, `reject` drops one from the
   * leaderboard, and `adopt` writes the current winner as a package now. Absent
   * writes no control request.
   */
  readonly control?: 'pause' | 'resume' | 'stop' | 'keep' | 'reject' | 'adopt'
  /**
   * The candidate `keep` and `reject` name, as the leaderboard lists it. Required
   * for those two: a person reading the table has an opinion about *that*
   * candidate, and a request that named none would be a guess.
   */
  readonly target?: string
  /**
   * With `keep` or `reject`, act on the candidate's descendants too, so a whole
   * branch is followed up or dropped rather than one point of it.
   */
  readonly lineage?: boolean
  /**
   * With `adopt`, the package name to write; the console's own default when
   * absent. Adopting never overwrites, so a name already in the workspace is
   * refused rather than replaced.
   */
  readonly as?: string
}

/** One mode a package declares, as the registry's catalogue lists it. */
export interface RefactoryPackageMode {
  /** The mode's name inside its package. */
  readonly name: string
  /** The mode's one-line description. */
  readonly description: string
  /** The package's trust level. */
  readonly trust: string
}

/** One package the registry knows, with the choices it declares. */
export interface RefactoryPackageChoice {
  /** The package name every registry verb accepts. */
  readonly name: string
  /**
   * Whether the package is enabled, or `null` when the registry reported no
   * modes for it: the state travels on each mode, so an empty package names none.
   */
  readonly enabled: boolean | null
  /** Where the package lives; empty when the registry cannot place it. */
  readonly where: string
  /** The modes the package declares. */
  readonly modes: readonly RefactoryPackageMode[]
}

/** What a graph spine last recorded about a run, as the console reports it. */
export interface RefactoryAuthoringProgress {
  /** The spine's own word: `done`, `ready`, `failed`, or `refused`. */
  readonly spine: string
  /** The nodes the spine left due, when it stopped with work outstanding. */
  readonly nodes_due: readonly string[]
  /** Why the spine refused or failed, as the spine wrote it. */
  readonly error: string
}

/** Task verdict counts, independent of the mode's aggregate scoring rule. */
export interface RefactoryItemResults {
  readonly count: number
  readonly passed: number
  readonly failed: number
  readonly unclassified: number
}

/** One mode used once: where it ran, how it ended, and what it recorded. */
export interface RefactorySingleRun {
  /** The run directory's name. */
  readonly run_id: string
  /** Absolute path of the run directory. */
  readonly root: string
  /** The mode the run was started from. */
  readonly mode: string | null
  /** One of `starting`, `running`, `finished`, `died`. */
  readonly state: string
  /** Whether the run wrote its report. */
  readonly finished: boolean
  /** Best score so far, or `null` when nothing scored yet. */
  readonly best_score: number | null
  /**
   * What the graph spine recorded, for a run the spine drove — a run that stopped
   * at a gate a person owns says so here — or `null` for a search run.
   */
  readonly authoring: RefactoryAuthoringProgress | null
  /** What the run wrote to `facts.json`, each value as text. */
  readonly facts: Readonly<Record<string, string>>
  /** Recorded Task item verdict counts, without assuming an aggregate score. */
  readonly item_results?: RefactoryItemResults | null
}

/** One run to start: the mode, where the run lives, and its settings. */
export interface RefactoryStartRequest {
  /** `<package>/<mode>`, as the catalogue lists it. */
  readonly mode: string
  /** Run id to claim; the console derives one when absent. */
  readonly runId?: string
  /** Directory holding the runs; the console default when absent. */
  readonly runsRoot?: string
  /** Run settings, passed through as the console's `--<name> <value>` flags. */
  readonly settings?: Readonly<Record<string, string | number | boolean>>
  /**
   * Knob values the search starts from, as `NAME=VALUE`, one `--knob` per element.
   * A mode whose inputs are frozen knobs (an input path, a command) needs one, or
   * every candidate runs the defaults and scores the same.
   */
  readonly knobs?: readonly string[]
}

/** A current one-shot invocation before its final answer. */
export interface RefactorySingleLaunch {
  readonly mode: string
  readonly root: string
  /** Prepared means a fresh durable invocation exists, not that execution succeeded. */
  readonly phase: 'starting' | 'prepared'
  readonly invocation?: string
}

/** Ordered launch and completion events from the same owned console call. */
export type RefactorySingleEvent =
  | { readonly kind: 'launch'; readonly launch: RefactorySingleLaunch }
  | { readonly kind: 'result'; readonly run: RefactorySingleRun }

/** One mode to use once: the mode, and the knob values it uses. */
export interface RefactorySingleRunRequest {
  /** Worker/model route and shared resource limits, passed to the same CLI flags. */
  readonly settings?: Readonly<Record<string, string | number>>
  /** `<package>/<mode>`, as the catalogue lists it. */
  readonly mode: string
  /** Knob values this run uses, as `NAME=VALUE`; the console's default when absent. */
  readonly knobs?: readonly string[]
}

/**
 * The Client Remote face over the console: every method the Host service marks
 * `@Remote`, with the carrier's outcome envelope.
 *
 * The harness's Typert generator emits this interface into `./remote` only when
 * its own protocol package is a project in the same workspace. The lean repo
 * depends on the published harness instead, so the generated artifacts are not
 * produced and this package states the face itself; the Host half that mounts it
 * still has to select this contribution in its own assembly.
 */
/** A registered project, as ``refactory projects`` lists them. */
export interface RefactoryProject {
  readonly name: string
  readonly path: string
  readonly kind: string
}

export interface RefactoryConsoleRemote {
  /** Live schemas and reviewed interface guidance. */
  help?: (topic: string, signal?: AbortSignal) => Promise<RemoteResult<Record<string, unknown>>>
  /** Read-only project worker defaults and provenance. */
  workers?: (signal?: AbortSignal) => Promise<RemoteResult<Record<string, unknown>>>
  /** Saved recipes; immutable versions differ from historical resolved snapshots. */
  profiles?: (signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryProfile[]>>
  /** Save an explicit new objective/settings version. */
  profileSave?: (name: string, spec: RefactoryProfileSpec, signal?: AbortSignal) => Promise<RemoteResult<RefactoryProfile>>
  /** Resolve the exact effective contract before launch. */
  profileResolve?: (request: RefactoryStartRequest, signal?: AbortSignal) => Promise<RemoteResult<Record<string, unknown>>>
  /** Includes failed and unmeasured candidates, not only eligible leaderboard rows. */
  comparison?: (run: string, signal?: AbortSignal) => Promise<RemoteResult<Record<string, unknown>>>
  /** @param signal - caller cancellation. @returns the configured console project directory. */
  project: (signal?: AbortSignal) => Promise<RemoteResult<string>>
  /** @param source - package source the mode comes from. @param name - mode package name. @param signal - caller cancellation. @returns the package list after the import. */
  importPackage: (source: string, name: string, signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryPackageChoice[]>>
  /** @param run - the run whose candidates are read. @param signal - caller cancellation. @returns the candidates by score, each with what a person did to it. */
  leaderboard: (run: string, signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryCandidateRow[]>>
  /** @param signal - caller cancellation. @returns the registered projects for target selection. */
  projects?: (signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryProject[]>>
  /** @param signal - caller cancellation. @returns the mode catalogue. */
  modes: (signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryModeChoice[]>>
  /** @param signal - caller cancellation. @returns the mode packages. */
  packages: (signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryPackageChoice[]>>
  /** @param name - package name to remove. @param signal - caller cancellation. @returns the package list after the removal. */
  removePackage: (name: string, signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryPackageChoice[]>>
  /** @param request - the mode to use once and its knob values. @param signal - caller cancellation. @returns the single run's answer. */
  run: (request: RefactorySingleRunRequest, signal?: AbortSignal) => Promise<RemoteResult<RefactorySingleRun>>
  /** @param request - one mode and its knobs. @param signal - cancels the owned console call. @returns launch identity, preparation evidence, then the final result. */
  runStream: (request: RefactorySingleRunRequest, signal: AbortSignal) => AsyncIterable<RefactorySingleEvent>
  /** @param signal - caller cancellation. @returns the runs the console knows about. */
  runs: (signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryRunSummary[]>>
  /** @param name - package name to switch. @param enabled - the state to record. @param signal - caller cancellation. @returns the package list after the change. */
  setPackageEnabled: (name: string, enabled: boolean, signal?: AbortSignal) => Promise<RemoteResult<readonly RefactoryPackageChoice[]>>
  /** @param request - the mode to start and its knob values. @param signal - caller cancellation. @returns the started run's receipt. */
  start: (request: RefactoryStartRequest, signal?: AbortSignal) => Promise<RemoteResult<RefactoryStartReceipt>>
  /** @param run - the run being read. @param after - byte offset already read. @param signal - caller cancellation. @returns the run's status frame. */
  status: (run: string, after: number, signal?: AbortSignal) => Promise<RemoteResult<RefactoryRunStatus>>
  graphControl?: (request: { run: string; action: string; node?: string; text?: string; revision?: string }, signal?: AbortSignal) => Promise<RemoteResult<Record<string, unknown>>>
  /** Observe without acquiring execution ownership or launching work. */
  watchStream?: (run: string, after: number, signal?: AbortSignal) => AsyncIterable<RefactoryWatchFrame>
  /** @param request - the run whose ledger is read, and what to append to it. @param signal - caller cancellation. @returns the ledger, and the gates the run is stopped at. */
  steer: (request: RefactorySteerRequest, signal?: AbortSignal) => Promise<RemoteResult<RefactorySteering>>
  /** @param run - the run to stop. @param signal - caller cancellation. @returns the stop receipt. */
  stop: (run: string, signal?: AbortSignal) => Promise<RemoteResult<RefactoryStopReceipt>>
  /** @param signal - caller cancellation. @returns the named outer loops, optimizers and evaluators. */
  strategies: (signal?: AbortSignal) => Promise<RemoteResult<RefactoryStrategies>>
}

/** One editable recipe, with immutable mode scoring and optional extra outputs. */
export interface RefactoryProfileSpec {
  readonly settings: Readonly<Record<string, unknown>>
  readonly contract: { readonly objective: string; readonly required_artifacts: readonly string[] }
}

/** A saved recipe version; launching writes a separate resolved snapshot. */
export interface RefactoryProfile extends RefactoryProfileSpec {
  readonly name: string
  readonly version: number
  readonly schema_version: number
}

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface RemoteErrorDetailsMap {
    /** `start` was asked for a mode the catalogue does not offer. */
    'refactory/unknown-mode': { readonly mode: string }
    /** `start` or `run` was asked for a mode whose own action is an authoring verb. */
    'refactory/authoring-mode': { readonly mode: string; readonly action: string }
    /** `start` or `run` was asked for a mode of a package the registry reports disabled. */
    'refactory/package-disabled': { readonly mode: string; readonly package: string }
    /**
     * The console refused the call, or answered with a shape it does not own.
     * `argv` is absent for a refusal the registry forwarded inside its own answer.
     */
    'refactory/console-failed': { readonly argv?: readonly string[] }
    /**
     * The console's collected stdout was cut by the collection cap before it
     * printed JSON, so there is no answer to decode. `capBytes` is the configured
     * cap and `retainedBytes` the length of the tail that survived; no
     * whole-stream total is claimed, because the process seam exposes none.
     */
    'refactory/console-truncated': {
      readonly argv: readonly string[]
      readonly capBytes: number
      readonly retainedBytes: number
    }
  }
}
