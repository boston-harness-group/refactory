/**
 * The picker's view state: what was read, what a person chose, and what the run
 * being read last reported.
 *
 * The choices are the Host's — a mode catalogue, three level-two registries and
 * the package list — and so is every status frame. What this store keeps is the
 * mount's own: which of the described choices is selected, the knob values a
 * person changed for it, the three chosen names, the run being read with its last
 * answer, and the run a mode was used once in. Bucketed by tab id, because two
 * picker tabs in one session choose independently.
 *
 * Writers run between `start` and `forget`: the owner's `signal` ends a bucket's
 * life, and the face stops dispatching once it aborts.
 */
import { defineStore, type EngineStoreHandle } from '@deepseek-ai/dsh-client-store'
import type {
  RefactoryCandidateRow,
  RefactoryKnob,
  RefactoryModeChoice,
  RefactoryPackageChoice,
  RefactoryProject,
  RefactoryRunStatus,
  RefactorySingleRun,
  RefactorySingleLaunch,
  RefactorySteering,
  RefactoryStrategies,
} from '@deepseek-ai/dsh-refactory-console/types'
import type { TabId } from '@deepseek-ai/dsh-client-ui-dockkit'

/** One read that either landed, is in flight, or failed with the Host's reason. */
export type Loadable<T> =
  | { readonly kind: 'loading' }
  | { readonly kind: 'ready'; readonly value: T }
  | {
    readonly kind: 'failed'
    readonly message: string
    /**
     * The Host's code and typed details, where the failure carries them.
     *
     * Both are optional because the message alone is what most failures have and
     * all of them show; they are kept so a read whose *kind* of failure the
     * reader must tell apart — a payload cut by the collection cap versus a
     * console that printed no JSON — can do it without matching text (R-5.4).
     */
    readonly code?: string
    readonly details?: unknown
  }

/** Everything a picker offers to choose between, read as one. */
export interface RefactoryChoices {
  /** Absolute configured console project, independent of the selected chat workspace. */
  readonly project: string
  /** The modes a run can be made from, each with the verb that runs it and its declared knobs. */
  readonly modes: readonly RefactoryModeChoice[]
  /** The named outer loops, optimizers and evaluators. */
  readonly strategies: RefactoryStrategies
  /** The packages the registry knows. */
  readonly packages: readonly RefactoryPackageChoice[]
  /** The registered projects for target selection. */
  readonly projects: readonly RefactoryProject[]
}

/** Search implementation choices and optional candidate evaluation limits. */
export interface RefactorySettings {
  /** The outer loop, from the `strategies` answer. */
  strategy: string
  /** The optimizer, from the `strategies` answer. */
  optimizer: string
  /** The evaluator, from the `strategies` answer. */
  evaluator: string
  /** Number of search epochs; absent keeps the console default. */
  epochs?: string
  /** Candidate proposals per epoch; absent keeps the console default. */
  batch_size?: string
  /** Maximum candidate evaluations; absent keeps the console default. */
  max_evaluations?: string
  /** Complete additional settings sent through the common JSON resolver. */
  configuration?: string
  /** Selected saved recipe and immutable version. */
  profile?: string
  profile_version?: string
}

/** Defaults of the console's LoopConfig, shown until a person changes a limit. */
export const SEARCH_BUDGET_DEFAULTS = { epochs: 2, batch_size: 4, max_evaluations: 64 } as const

/** The existing console settings exposed by the search budget fields. */
export type RefactoryBudgetName = keyof typeof SEARCH_BUDGET_DEFAULTS

/**
 * Parse only explicitly entered limits; missing values retain console defaults.
 * @param settings - the tab's choices, including in-progress numeric input.
 * @returns admitted positive integers, or null when any edited field is invalid.
 */
export function searchBudget(settings: Pick<RefactorySettings, RefactoryBudgetName>): Partial<Record<RefactoryBudgetName, number>> | null {
  const budget: Partial<Record<RefactoryBudgetName, number>> = {}
  for (const name of Object.keys(SEARCH_BUDGET_DEFAULTS) as RefactoryBudgetName[]) {
    const value = settings[name]
    if (value === undefined) continue
    const parsed = Number(value)
    if (value.trim() === '' || !Number.isSafeInteger(parsed) || parsed < 1) return null
    budget[name] = parsed
  }
  return budget
}

/**
 * How much of a watched run's journalled history the panel shows.
 *
 * Exactly these three, and `standard` is the default (INT-001, DEC-002): the
 * level is a rendering choice, so it never reaches the console and never filters
 * what is read — the tab holds the same records at every level and only decides
 * how many of them it draws.
 */
export type RefactoryActivityLevel = 'compact' | 'standard' | 'verbose'

/** The level a fresh tab reads at, and the only one the spec names as the default. */
export const DEFAULT_ACTIVITY_LEVEL: RefactoryActivityLevel = 'standard'

/** The three levels, in the order the control offers them. */
export const ACTIVITY_LEVELS: readonly RefactoryActivityLevel[] = ['compact', 'standard', 'verbose']

/**
 * How many held records are in the DOM at once.
 *
 * Rendering is bounded, holding is not: a verbose pass holds every record, and
 * drawing one `<li>` per record would block the tab on a large run. The window
 * equals one console page so a page arrival and a scroll move it the same way.
 */
export const RENDER_WINDOW = 200

/** One page of a run's journal, as the reader that fetched it reports it. */
export interface RefactoryActivityPage {
  /** Journal index the page's first event sits at. */
  readonly after: number
  /** Index of the next record: what a later read continues from. */
  readonly cursor: number
  /** The page's records, in journal order. */
  readonly events: readonly Readonly<Record<string, unknown>>[]
  /** Complete records in the journal as this page's answer counted them. */
  readonly total: number
  /** Whether the console pages the history at all (false for an older console). */
  readonly paged: boolean
  /** The run this page belongs to, when the answer names one. */
  readonly run: string | null
}

/**
 * The journalled history a tab holds for the run it is reading.
 *
 * `events` is deliberately uncapped. Capping it would silently turn verbose into
 * "the last N", which is the one thing DEC-001 forbids; what is bounded is the
 * slice drawn (`RENDER_WINDOW`). `complete` is the reader's evidence that the
 * history is whole — it is set only by an answer that reported `has_more: false`,
 * and never by the absence of an error.
 */
export interface HeldHistory {
  /** The run these records belong to (R-4.5). */
  readonly run: string
  /** Held records, oldest first, uncapped. */
  readonly events: readonly Readonly<Record<string, unknown>>[]
  /** The journal index `events[0]` sits at: 0 once the history reaches the start. */
  readonly first: number
  /** Highest journal cursor held: where the next forward read continues. */
  readonly through: number
  /** Last reported journal total. */
  readonly total: number
  /** An answer with `has_more: false` has been seen for this run. */
  readonly complete: boolean
  /** False when the console does not page (R-6.3). */
  readonly paged: boolean
  /** A page is in flight. */
  readonly reading: boolean
  /** Why the last page failed, or `null`. */
  readonly error: string | null
}

/** Why a gesture was refused: the Host's code, and the Host's own message. */
export interface RefactoryNotice {
  /** The Remote failure's code, e.g. `refactory/authoring-mode`. */
  readonly code: string
  /** The Host's message, shown as data when the code names nothing more precise. */
  readonly message: string
}

/**
 * The one-shot use of a mode: still going, or what it left.
 *
 * A refusal is not a state here — it is the gesture's notice, where the code still
 * has the copy that names what to do instead — so a result is only ever the run
 * that came back.
 */
export type RefactorySingleState =
  | { readonly kind: 'running'; readonly launch?: RefactorySingleLaunch }
  | { readonly kind: 'failed'; readonly launch: RefactorySingleLaunch }
  | { readonly kind: 'ready'; readonly value: RefactorySingleRun }

/** One tab's picker: what it read, what it chose, and where the run got to. */
export interface RefactoryTabState {
  /** The choices, as last read. */
  choices: Loadable<RefactoryChoices>
  /** The chosen mode's catalogue id, or `null` before anything is chosen. */
  chosen: string | null
  /** Package filter; selection never changes the watched run. */
  selectedPackage: string | null
  /** Presentation intent survives dock remounts, without changing execution settings. */
  setupIntent: 'run' | 'optimize'
  /**
   * The knob values a person changed for the chosen mode, by knob name. A knob
   * that is absent keeps the value its own declaration gives it.
   */
  knobs: Record<string, string>
  /** The chosen level-two names. */
  settings: RefactorySettings
  /** The run being read, or `null` when nothing is being watched. */
  watching: string | null
  /** The last status frame for `watching`; `null` before the first read. */
  frame: Loadable<RefactoryRunStatus> | null
  /** The journal history held for `watching`; `null` before anything is held. */
  activity: HeldHistory | null
  /**
   * The chosen verbosity level. Per tab and never persisted across tab records:
   * the level belongs to the record, not to the tab type or to the session key.
   */
  activityLevel: RefactoryActivityLevel
  /** The row index of the first held record the history window draws. */
  activityScroll: number
  /** The watched run's intent ledger and gates; `null` before the ledger is read. */
  steering: Loadable<RefactorySteering> | null
  /** The watched run's candidates, with what a person did to each; `null` before they are read. */
  candidates: Loadable<readonly RefactoryCandidateRow[]> | null
  /** The last one-shot request and what it left; independent of the current selection. */
  single: RefactorySingleState | null
  /** What the last refused gesture said, or `null`. */
  notice: RefactoryNotice | null
  /** Whether a package change is in flight. */
  busy: boolean
  /** Whether an optimization launch is awaiting its receipt. */
  starting: boolean
}

/** Every tab's picker, keyed by tab id. */
export interface RefactoryState {
  byTab: Record<TabId, RefactoryTabState>
}

/** Named search implementations and candidate evaluation limits. */
export type RefactorySettingName = keyof RefactorySettings

/**
 * One tab's bucket, which every writer after `start` relies on: the face only
 * dispatches while the record's signal is live, and `forget` runs on its abort.
 * @param state - the draft.
 * @param tabId - the tab being written.
 * @returns the tab's picker.
 */
function bucket(state: RefactoryState, tabId: TabId): RefactoryTabState {
  const tab = state.byTab[tabId]
  if (tab === undefined) throw new Error(`ui-refactory: no picker for tab "${tabId}"`)
  return tab
}

/** The picker store's write set; every action names the tab it writes. */
type RefactoryActions = {
  start: (draft: RefactoryState, tabId: TabId) => void
  loading: (draft: RefactoryState, tabId: TabId) => void
  loaded: (draft: RefactoryState, tabId: TabId, choices: RefactoryChoices) => void
  failed: (draft: RefactoryState, tabId: TabId, message: string) => void
  chosePackage: (draft: RefactoryState, tabId: TabId, name: string | null) => void
  setupIntent: (draft: RefactoryState, tabId: TabId, intent: 'run' | 'optimize') => void
  chose: (draft: RefactoryState, tabId: TabId, id: string | null) => void
  knob: (draft: RefactoryState, tabId: TabId, name: string, value: string) => void
  setting: (draft: RefactoryState, tabId: TabId, name: RefactorySettingName, value: string) => void
  watching: (draft: RefactoryState, tabId: TabId, run: string) => void
  frameLoading: (draft: RefactoryState, tabId: TabId) => void
  framed: (draft: RefactoryState, tabId: TabId, frame: RefactoryRunStatus) => void
  frameFailed: (draft: RefactoryState, tabId: TabId, message: string, code?: string, details?: unknown) => void
  setActivityLevel: (draft: RefactoryState, tabId: TabId, level: RefactoryActivityLevel) => void
  activityPage: (draft: RefactoryState, tabId: TabId, run: string, page: RefactoryActivityPage) => void
  activityEarlier: (draft: RefactoryState, tabId: TabId, run: string, page: RefactoryActivityPage) => void
  activityPageLoading: (draft: RefactoryState, tabId: TabId, run: string) => void
  activityPageFailed: (draft: RefactoryState, tabId: TabId, run: string, message: string) => void
  activityScrolled: (draft: RefactoryState, tabId: TabId, row: number) => void
  steeringLoading: (draft: RefactoryState, tabId: TabId) => void
  steered: (draft: RefactoryState, tabId: TabId, steering: RefactorySteering) => void
  steeringFailed: (draft: RefactoryState, tabId: TabId, message: string) => void
  candidatesLoading: (draft: RefactoryState, tabId: TabId) => void
  candidatesRead: (draft: RefactoryState, tabId: TabId, candidates: readonly RefactoryCandidateRow[]) => void
  candidatesFailed: (draft: RefactoryState, tabId: TabId, message: string) => void
  runLoading: (draft: RefactoryState, tabId: TabId) => void
  runLaunched: (draft: RefactoryState, tabId: TabId, launch: RefactorySingleLaunch) => void
  ran: (draft: RefactoryState, tabId: TabId, run: RefactorySingleRun) => void
  runFailed: (draft: RefactoryState, tabId: TabId) => void
  packages: (draft: RefactoryState, tabId: TabId, packages: readonly RefactoryPackageChoice[]) => void
  noticed: (draft: RefactoryState, tabId: TabId, notice: RefactoryNotice | null) => void
  busy: (draft: RefactoryState, tabId: TabId, busy: boolean) => void
  starting: (draft: RefactoryState, tabId: TabId, starting: boolean) => void
  forget: (draft: RefactoryState, tabId: TabId) => void
}

/** The level-two names honestly absent, so a run takes the console's defaults for them. */
export const NO_SETTINGS: RefactorySettings = { strategy: '', optimizer: '', evaluator: '' }

/** One knob value as a gesture sends it. */
export interface RefactoryKnobSetting {
  /** The knob's declared name. */
  readonly name: string
  /** The value to send, as text. */
  readonly value: string
}

/** One declared knob, with the value this run gives it. */
export interface RefactoryKnobValue extends RefactoryKnobSetting {
  /** The declaration, as the catalogue reports it. */
  readonly knob: RefactoryKnob
}

/**
 * Every knob the chosen mode declares, with what a person changed and each
 * declaration's own value for the rest.
 *
 * This is what a gesture sends, and both gestures send the same one: a knob a
 * person never touched still travels, so a run is the one the fields showed rather
 * than the one some other defaulting would have made.
 * @param mode - the chosen catalogue entry, or `undefined` when none is chosen.
 * @param edited - the values a person changed, by knob name; absent before anything is typed.
 * @returns one entry per declared knob, in declaration order.
 */
export function knobValues(
  mode: RefactoryModeChoice | undefined,
  edited: Readonly<Record<string, string>> | undefined,
): readonly RefactoryKnobValue[] {
  if (mode === undefined) return []
  return Object.entries(mode.parameters).map(([name, knob]) => ({
    name,
    knob,
    value: edited?.[name] ?? knob.value,
  }))
}

/**
 * Declare the picker's store.
 *
 * A factory rather than a shared handle: the registration declares it as an
 * exclusive store, so the framework mints one instance per session.
 * @returns the store handle to declare on the registration.
 */
export function createRefactoryStore(): EngineStoreHandle<RefactoryState, RefactoryActions> {
  return defineStore({
    init: (): RefactoryState => ({ byTab: {} }),
    actions: {
      /**
       * Seed one tab's picker, with the choices already being read.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      start: (d, tabId: TabId) => {
        d.byTab[tabId] = {
          choices: { kind: 'loading' },
          chosen: null,
          selectedPackage: null,
          setupIntent: 'run',
          knobs: {},
          settings: { ...NO_SETTINGS },
          watching: null,
          frame: null,
          activity: null,
          activityLevel: DEFAULT_ACTIVITY_LEVEL,
          activityScroll: 0,
          steering: null,
          candidates: null,
          single: null,
          notice: null,
          busy: false,
          starting: false,
        }
      },
      /**
       * Mark the choices as being read again, keeping what a person chose.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      loading: (d, tabId: TabId) => {
        bucket(d, tabId).choices = { kind: 'loading' }
      },
      /**
       * Record the choices the Host answered with.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param choices - the modes, strategies and packages.
       */
      loaded: (d, tabId: TabId, choices: RefactoryChoices) => {
        bucket(d, tabId).choices = { kind: 'ready', value: choices }
      },
      /**
       * Record why the choices could not be read, and drop a selection they may
       * no longer contain.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param message - the Host's reason.
       */
      failed: (d, tabId: TabId, message: string) => {
        const tab = bucket(d, tabId)
        tab.choices = { kind: 'failed', message }
        tab.chosen = null
        tab.knobs = {}
      },
      /**
       * Choose one of the described modes, or nothing, and start its knobs at the
       * values they declare.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param id - the chosen mode's catalogue id, or `null`.
       */
      chosePackage: (d, tabId: TabId, name: string | null) => {
        const tab = bucket(d, tabId)
        if (tab.selectedPackage === name) return
        tab.selectedPackage = name
        tab.chosen = null
        const savedTarget = tab.knobs.target
        tab.knobs = {}
        if (savedTarget) tab.knobs.target = savedTarget
      },
      setupIntent: (d, tabId: TabId, intent: 'run' | 'optimize') => { bucket(d, tabId).setupIntent = intent },
      chose: (d, tabId: TabId, id: string | null) => {
        const tab = bucket(d, tabId)
        tab.chosen = id
        if (id && tab.choices.kind === 'ready') tab.selectedPackage = tab.choices.value.modes.find(mode => mode.id === id)?.package ?? null
        // Knobs belong to the selection. A one-shot belongs to its own request
        // and keeps its result even when the person selects another mode.
        // The target knob is preserved because it is set by the target picker
        // above the mode select and should not be silently dropped.
        const savedTarget = tab.knobs.target
        tab.knobs = {}
        if (savedTarget) tab.knobs.target = savedTarget
      },
      /**
       * Record one knob value a person changed.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param name - the knob's declared name.
       * @param value - the value to send, as text.
       */
      knob: (d, tabId: TabId, name: string, value: string) => {
        bucket(d, tabId).knobs[name] = value
      },
      /**
       * Choose one of the described level-two names.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param name - which seam the choice is for.
       * @param value - the chosen name, or the empty string for the default.
       */
      setting: (d, tabId: TabId, name: RefactorySettingName, value: string) => {
        bucket(d, tabId).settings[name] = value
      },
      /**
       * Begin reading one run's status.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run id or directory the Host reported.
       */
      watching: (d, tabId: TabId, run: string) => {
        const tab = bucket(d, tabId)
        // A different run is a different history: dropping it is what keeps two
        // runs' journal indices from being spliced into one list (R-4.4). The
        // level survives, because which detail a person wants is about them and
        // not about the run (R-2.1).
        tab.activity = null
        tab.activityScroll = 0
        tab.watching = run
        tab.frame = { kind: 'loading' }
        tab.steering = { kind: 'loading' }
        tab.candidates = null
        tab.notice = null
      },
      /**
       * Mark the watched run's status as being read.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      frameLoading: (d, tabId: TabId) => {
        bucket(d, tabId).frame = { kind: 'loading' }
      },
      /**
       * Record one status frame.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param frame - where the run is.
       */
      framed: (d, tabId: TabId, frame: RefactoryRunStatus) => {
        bucket(d, tabId).frame = { kind: 'ready', value: frame }
      },
      /**
       * Replace the packages with the catalogue a package gesture answered with.
       *
       * A package change cannot alter a mode's own description or the level-two
       * registries, so the modes and strategies already read stay as they are; a
       * gesture before the first read lands is dropped, because that read is
       * already fetching the packages it would replace.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param packages - the packages the Host answered with.
       */
      packages: (d, tabId: TabId, packages: readonly RefactoryPackageChoice[]) => {
        const tab = bucket(d, tabId)
        const choices = tab.choices
        if (choices.kind !== 'ready') return
        tab.choices = { kind: 'ready', value: { ...choices.value, packages } }
      },
      /**
       * Record why the watched run's status could not be read.
       *
       * The failure belongs to the status the reader is looking at, so it is
       * written where that status is rather than as a notice about a gesture.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param message - the Host's reason.
       */
      frameFailed: (d, tabId: TabId, message: string, code?: string, details?: unknown) => {
        bucket(d, tabId).frame = {
          kind: 'failed',
          message,
          ...code === undefined ? {} : { code },
          ...details === undefined ? {} : { details },
        }
      },
      /**
       * Choose how much of the held history to draw.
       *
       * Only the level moves: the records already held stay held, so a level
       * switch neither refetches a page nor loses the run's summary (R-2.7), and
       * the window starts at the newest record again because that is where a
       * person arriving at a level looks first.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param level - the level to draw at.
       */
      setActivityLevel: (d, tabId: TabId, level: RefactoryActivityLevel) => {
        const tab = bucket(d, tabId)
        tab.activityLevel = level
        tab.activityScroll = 0
      },
      /**
       * Splice one forward page onto the held history.
       *
       * The page goes wherever its `after` says it belongs rather than always at
       * the end: a polled page starts at the held cursor, so appending it keeps
       * journal order, while a page-0 refetch after a gap replaces the indices it
       * covers. Splicing by index is also what makes the two cases one write.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run the page belongs to.
       * @param page - the page the console answered with.
       */
      activityPage: (d, tabId: TabId, run: string, page: RefactoryActivityPage) => {
        const tab = bucket(d, tabId)
        const held = tab.activity?.run === run ? tab.activity : null
        // A splice is only safe where the page's own index is what the held list
        // already means. `after` past the held end would leave a hole, and
        // anything else is a page for history this tab does not hold (R-4.5).
        const after = page.paged ? Math.max(0, page.after) : 0
        if (held !== null && after > held.events.length) return
        const events = held === null ? [] : [...held.events]
        if (page.paged) events.splice(after, events.length - after, ...page.events)
        else events.splice(0, events.length, ...page.events)
        tab.activity = {
          run,
          events,
          // The list still begins where it began: `after` says where this page
          // goes, not where the history starts.
          first: page.paged ? held?.first ?? after : 0,
          through: page.paged ? Math.max(after + page.events.length, page.cursor) : page.events.length,
          total: page.paged ? page.total : page.events.length,
          // An unpaged console answered with its whole history, so the answer
          // itself is the completeness evidence a `has_more: false` would be.
          complete: !page.paged || page.cursor >= page.total,
          paged: page.paged,
          reading: false,
          error: null,
        }
      },
      /**
       * Put an older page in front of the held history.
       *
       * "Load earlier activity" is the one gesture that moves the window
       * backwards, so the retained scroll row shifts by exactly the records that
       * were added ahead of it — the row a person was reading stays under them.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run the page belongs to.
       * @param page - the earlier page the console answered with.
       */
      activityEarlier: (d, tabId: TabId, run: string, page: RefactoryActivityPage) => {
        const tab = bucket(d, tabId)
        const held = tab.activity?.run === run ? tab.activity : null
        if (held === null || !page.paged) return
        // A backwards read may overlap what is held: the page ends at its own
        // `cursor`, and the held list begins at `held.first`. Only the records
        // before that first one are new; the rest would duplicate rows.
        const prefix = page.events.slice(0, Math.max(0, Math.min(page.events.length, held.first - page.after)))
        tab.activity = {
          ...held,
          events: [...prefix, ...held.events],
          first: Math.max(0, Math.min(held.first, page.after)),
          through: held.through,
          total: page.total,
          complete: held.complete,
          reading: false,
          error: null,
        }
        tab.activityScroll = tab.activityScroll + prefix.length
      },
      /**
       * Say that a page is on its way for one run.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run the page belongs to.
       */
      activityPageLoading: (d, tabId: TabId, run: string) => {
        const tab = bucket(d, tabId)
        const held = tab.activity?.run === run ? tab.activity : null
        tab.activity = held === null
          ? { run, events: [], first: 0, through: 0, total: 0, complete: false, paged: true, reading: true, error: null }
          : { ...held, reading: true, error: null }
      },
      /**
       * Record why a page could not be read, keeping every record already held.
       *
       * A failed page is not a lost history: what was read stays drawn and the
       * failure is a line beside it, because throwing the held records away
       * would make a transient console refusal look like a shorter journal.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run the page belongs to.
       * @param message - the Host's reason.
       */
      activityPageFailed: (d, tabId: TabId, run: string, message: string) => {
        const tab = bucket(d, tabId)
        const held = tab.activity?.run === run ? tab.activity : null
        tab.activity = held === null
          ? { run, events: [], first: 0, through: 0, total: 0, complete: false, paged: true, reading: false, error: message }
          : { ...held, reading: false, error: message }
      },
      /**
       * Remember where the history window is, as a row index in the held list.
       *
       * A row rather than a pixel offset: a page arrival changes the list under a
       * scroll, and a row keeps meaning the same record where a pixel count does
       * not.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param row - the row index of the first drawn record.
       */
      activityScrolled: (d, tabId: TabId, row: number) => {
        bucket(d, tabId).activityScroll = Math.max(0, Math.floor(row))
      },
      /**
       * Mark the watched run's ledger as being read.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      steeringLoading: (d, tabId: TabId) => {
        bucket(d, tabId).steering = { kind: 'loading' }
      },
      /**
       * Record one read of the watched run's ledger and gates.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param steering - the ledger, and the gates the run is stopped at.
       */
      steered: (d, tabId: TabId, steering: RefactorySteering) => {
        bucket(d, tabId).steering = { kind: 'ready', value: steering }
      },
      /**
       * Record why the watched run's ledger could not be read.
       *
       * Like a status failure, this belongs where the ledger is drawn rather than as
       * a notice about a gesture: the reader is looking at the ledger.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param message - the Host's reason.
       */
      steeringFailed: (d, tabId: TabId, message: string) => {
        bucket(d, tabId).steering = { kind: 'failed', message }
      },
      /**
       * Mark the watched run's candidates as being read.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      candidatesLoading: (d, tabId: TabId) => {
        bucket(d, tabId).candidates = { kind: 'loading' }
      },
      /**
       * Record one read of the watched run's candidates.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param candidates - the rows, highest score first.
       */
      candidatesRead: (d, tabId: TabId, candidates: readonly RefactoryCandidateRow[]) => {
        bucket(d, tabId).candidates = { kind: 'ready', value: candidates }
      },
      /**
       * Record why the watched run's candidates could not be read.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param message - the Host's reason.
       */
      candidatesFailed: (d, tabId: TabId, message: string) => {
        bucket(d, tabId).candidates = { kind: 'failed', message }
      },
      /**
       * Mark the chosen mode as being used once.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      runLoading: (d, tabId: TabId) => {
        bucket(d, tabId).single = { kind: 'running' }
      },
      /**
       * Record what using the chosen mode once left.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param run - the run, with where it got to.
       */
      runLaunched: (d, tabId: TabId, launch: RefactorySingleLaunch) => {
        bucket(d, tabId).single = { kind: 'running', launch }
      },
      ran: (d, tabId: TabId, run: RefactorySingleRun) => {
        bucket(d, tabId).single = { kind: 'ready', value: run }
      },
      /**
       * Stop waiting on the one-shot run: it was refused, so there is no run to
       * show and the refusal is the gesture's notice.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       */
      runFailed: (d, tabId: TabId) => {
        const tab = bucket(d, tabId)
        const launch = tab.single?.kind === 'running' ? tab.single.launch : undefined
        tab.single = launch === undefined ? null : { kind: 'failed', launch }
      },
      /**
       * Say what the last refused gesture was.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param notice - the failure to show, or `null` to clear it.
       */
      noticed: (d, tabId: TabId, notice: RefactoryNotice | null) => {
        bucket(d, tabId).notice = notice
      },
      /**
       * Record whether a package change is in flight.
       * @param d - draft state.
       * @param tabId - the tab being drawn.
       * @param busy - whether the change is still running.
       */
      starting: (d, tabId: TabId, starting: boolean) => { bucket(d, tabId).starting = starting },
      busy: (d, tabId: TabId, busy: boolean) => {
        bucket(d, tabId).busy = busy
      },
      /**
       * Forget one tab's picker, for a tab record that is gone.
       * @param d - draft state.
       * @param tabId - the tab that went away.
       */
      forget: (d, tabId: TabId) => {
        d.byTab = Object.fromEntries(Object.entries(d.byTab).filter(([id]) => id !== tabId))
      },
    },
  })
}
