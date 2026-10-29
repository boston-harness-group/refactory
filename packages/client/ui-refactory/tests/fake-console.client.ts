/**
 * A scripted `refactoryConsole` Remote face, and the described choices a spec
 * feeds it.
 *
 * Every call resolves from the spec's own value, so a spec states what the Host
 * answered with and reads what the picker did with it. A refusal is built from
 * a code and a message: the picker reads exactly those two fields, and the
 * carrier's error class carries more than a picker test needs.
 */
import { vi } from 'vitest'
import type { Mock } from 'vitest'
import type { RemoteFailure, RemoteResult } from '@deepseek-ai/dsh-api-remotes/client'
import type {
  RefactoryActivity,
  RefactoryCandidateRow,
  RefactoryModeChoice,
  RefactoryPackageChoice,
  RefactoryProject,
  RefactoryRunStatus,
  RefactorySingleRun,
  RefactoryStartReceipt,
  RefactorySteering,
  RefactoryStrategies,
} from '@deepseek-ai/dsh-refactory-console/types'
import type { RefactoryRemote } from '../src/client/face.ts'

/** The three registries with nothing in them. */
export const NO_STRATEGIES: RefactoryStrategies = { outer_loops: [], optimizers: [], evaluators: [] }

/** The knobs one catalogue entry declares: one of every kind a panel draws. */
export const PARAMETERS: RefactoryModeChoice['parameters'] = {
  rounds: {
    value: '3', kind: 'int', values: [], range: [1, 10], frozen: false, description: 'how many rounds',
  },
  temperature: {
    value: '0.5', kind: 'float', values: [], range: [null, null], frozen: false, description: '',
  },
  source_root: {
    value: '/tmp/src', kind: 'str', values: [], range: [null, null], frozen: true, description: 'the tree to patch',
  },
  preset: {
    value: 'broad', kind: 'str', values: ['broad', 'narrow'], range: [null, null], frozen: false, description: 'the search preset',
  },
  verbose: {
    value: 'false', kind: 'bool', values: [], range: [null, null], frozen: false, description: 'say more',
  },
}

/** A design-mode parameter set that includes the ``target`` knob. */
export const DESIGN_PARAMETERS: RefactoryModeChoice['parameters'] = {
  ...PARAMETERS,
  target: {
    value: '', kind: 'str', values: [], range: [null, null], frozen: true, description: 'the git repository to change',
  },
}

/** One scored mode, as the catalogue describes it. */
export const MODE: RefactoryModeChoice = {
  id: 'p/adaptive',
  package: 'p',
  mode: 'adaptive',
  description: 'search the modes',
  startable: true,
  enabled: true,
  scored: true,
  action: 'start',
  trust: 'trusted',
  parameters: PARAMETERS,
}

/** A design mode that declares ``target`` and shows the target picker. */
export const DESIGN: RefactoryModeChoice = {
  id: 'default_pack/design',
  package: 'default_pack',
  mode: 'design',
  description: 'design and build an artifact',
  startable: true,
  enabled: true,
  scored: false,
  action: 'design',
  trust: 'trusted',
  parameters: DESIGN_PARAMETERS,
}

/** One authoring mode, whose `action` is the verb that runs it. */
export const AUTHORING: RefactoryModeChoice = {
  ...MODE,
  id: 'default_pack/create',
  package: 'default_pack',
  mode: 'create',
  description: 'create a package of modes',
  scored: false,
  action: 'create',
}

/** One mode whose package is disabled, so the loop cannot run it here. */
export const DISABLED: RefactoryModeChoice = {
  ...MODE,
  id: 'off/adaptive',
  package: 'off',
  enabled: false,
  startable: false,
}

/** One package with one mode. */
export const PACKAGE: RefactoryPackageChoice = {
  name: 'p',
  enabled: true,
  where: '/repo/.factory/packages/p',
  modes: [{ name: 'adaptive', description: 'search the modes', trust: 'trusted' }],
}

/** The three registries, each naming one implementation. */
export const STRATEGIES: RefactoryStrategies = {
  outer_loops: [{ name: 'beam', description: 'keep the best few' }],
  optimizers: [{ name: 'random', description: 'sample uniformly' }],
  evaluators: [{ name: 'score', description: 'the mode\'s own score' }],
}

/** One status frame, mid-run. */
export const FRAME: RefactoryRunStatus = {
  kind: 'optimizer',
  run_id: 'live',
  root: '/runs/live',
  mode: 'p/adaptive',
  state: 'running',
  events: [],
  cursor: 3,
  counters: { epoch: 0, generation: 1, evaluations: 2, steps: 1 },
  best_score: 1.25,
  finished: false,
  running: true,
}

/** A run's intent ledger, with no gate waiting. */
export const STEERING: RefactorySteering = {
  run: '/runs/live',
  ledger: '/runs/live/intent-ledger.md',
  exists: true,
  entries: [
    { at: '2026-01-01T00:00:00+00:00', kind: 'Initial Idea', text: 'make it better' },
    { at: '2026-01-01T00:01:00+00:00', kind: 'correction', text: 'use a smaller grid' },
  ],
  constraints: [],
  verdicts: {},
  gates: [],
  problems: [],
}

/** One ledger read that answered with a note the person just added. */
export const STEERED: RefactorySteering = {
  ...STEERING,
  entries: [...STEERING.entries, { at: '2026-01-01T00:02:00+00:00', kind: 'info', text: 'try the other seed' }],
}

/** A run stopped at a `user` gate, with the verdicts the gate accepts. */
export const GATED_STEERING: RefactorySteering = {
  ...STEERING,
  gates: [{ id: 'approve', prompt: 'the person approves the draft', verdicts: ['proceed', 'halt'] }],
}

/** What the console recorded about a start. */
export const RECEIPT: RefactoryStartReceipt = {
  run_id: 'live',
  mode: 'p/adaptive',
  root: '/runs/live',
  started: 1.5,
  config: { epochs: '1' },
  log: 'live.log',
  pid: 7,
}

/** One mode used once: a search-shaped run with no graph spine. */
export const SINGLE: RefactorySingleRun = {
  run_id: 'p-adaptive-20260101-000000',
  root: '/runs/single',
  mode: 'p/adaptive',
  state: 'finished',
  finished: true,
  best_score: 0.75,
  authoring: null,
  facts: { name: 'demo', validation: 'ok' },
}

/**
 * Two candidates as the console lists them: the leader, and one a person has
 * already dismissed, so a spec can draw the flags the controls read.
 */
export const CANDIDATES: readonly RefactoryCandidateRow[] = [
  { id: 'aaaa11', parent: null, score: 0.96, raw_score: 1.0, nodes: 4, kept: false, rejected: false },
  { id: 'bbbb22', parent: 'aaaa11', score: 0.5, raw_score: 0.6, nodes: 5, kept: true, rejected: false },
  { id: 'cccc33', parent: 'aaaa11', score: 0.4, raw_score: 0.5, nodes: 6, kept: false, rejected: true },
]

/** One mode used once that stopped at a gate a person owns. */
export const GATED: RefactorySingleRun = {
  ...SINGLE,
  state: 'running',
  finished: false,
  best_score: null,
  authoring: { spine: 'ready', nodes_due: ['approve', 'overwatch'], error: '' },
}

/** One mode used once whose graph finished, with nothing recorded to show. */
export const COMPLETED: RefactorySingleRun = {
  ...SINGLE,
  authoring: { spine: 'done', nodes_due: [], error: '' },
  facts: {},
}

/**
 * A run's activity with its journal paged: `after`..`cursor` of `total` records.
 *
 * `has_more` is computed the way the console computes it (`cursor < total`), so a
 * spec that feeds pages to a reader is feeding it the same identity the console
 * promises (R-1.4) rather than a fixed flag.
 * @param after - the journal index the page starts at.
 * @param limit - how many records this answer may carry.
 * @param total - complete records in the journal.
 * @param run - the run the answer names.
 * @returns the activity envelope, with one placeholder record per index.
 */
export function ACTIVITY(after: number, limit: number, total: number, run = '/runs/live'): RefactoryActivity {
  const cursor = Math.min(total, after + limit)
  return {
    schema_version: 1,
    run_id: run,
    invocation: 'inv-1',
    cursor,
    after,
    total,
    page_size: limit,
    has_more: cursor < total,
    last_event_at: '2026-01-01T00:00:00+00:00',
    started_at: '2026-01-01T00:00:00+00:00',
    nodes: ['build'],
    completed: [],
    graph: null,
    wait_reason: null,
    workers: [],
    deliveries: [],
    events: Array.from({ length: cursor - after }, (_unused, index) => ({
      at: `t${after + index}`, kind: 'worker_event', node_id: 'build',
    })),
    outputs: { name: 'demo' },
    resources: null,
    control: null,
  }
}

/** A landed Remote call. */
export function ok<T>(value: T): RemoteResult<T> {
  return { ok: true, value }
}

/** A refused Remote call, carrying the code and message the picker reads. */
export function failure<T>(code: string, message: string): RemoteResult<T> {
  return { ok: false, error: { code, message } as unknown as RemoteFailure }
}

/** One project, as the console lists them. */
export const PROJECT: RefactoryProject = {
  name: 'weather-cli',
  path: '/projects/weather-cli',
  kind: 'new',
}

/** The scripted Remote face and the mocks behind it. */
export interface FakeConsole {
  readonly remote: RefactoryRemote
  readonly modes: Mock
  readonly project: Mock
  readonly projects: Mock
  readonly strategies: Mock
  readonly packages: Mock
  readonly start: Mock
  readonly run: Mock
  readonly status: Mock
  readonly leaderboard: Mock
  readonly steer: Mock
  readonly stop: Mock
  readonly setPackageEnabled: Mock
  readonly removePackage: Mock
  readonly importPackage: Mock
}

/**
 * Build a Remote face whose every method resolves from a settable answer.
 * @returns the face and its mocks.
 */
export function fakeConsole(): FakeConsole {
  const methods = {
    modes: vi.fn(async () => ok<readonly RefactoryModeChoice[]>([])),
    project: vi.fn(async () => ok('/repo')),
    projects: vi.fn(async () => ok<readonly RefactoryProject[]>([])),
    strategies: vi.fn(async () => ok(NO_STRATEGIES)),
    packages: vi.fn(async () => ok<readonly RefactoryPackageChoice[]>([])),
    start: vi.fn(async () => ok(RECEIPT)),
    run: vi.fn(async (_request?: unknown, _signal?: AbortSignal) => ok(SINGLE)),
    status: vi.fn(async () => ok(FRAME)),
    leaderboard: vi.fn(async () => ok<readonly RefactoryCandidateRow[]>(CANDIDATES)),
    steer: vi.fn(async () => ok(STEERING)),
    runs: vi.fn(async () => ok([])),
    stop: vi.fn(async () => ok({ stopped: true })),
    setPackageEnabled: vi.fn(async () => ok<readonly RefactoryPackageChoice[]>([])),
    removePackage: vi.fn(async () => ok<readonly RefactoryPackageChoice[]>([])),
    importPackage: vi.fn(async () => ok<readonly RefactoryPackageChoice[]>([])),
  }
  return { remote: { refactoryConsole: { ...methods, runStream: async function* (request, signal) {
    const result = await methods.run(request, signal)
    if (!result.ok) throw new Error(result.error.message)
    yield { kind: 'result' as const, run: result.value }
  } } }, ...methods }
}

/** Let every settled Remote promise's `.then` run. */
export async function settle(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}
