/**
 * The interactive workflow runner: a per-session {@link Spine} plus the
 * `wf_*` tools the CEO uses to walk the graph step by step, and the
 * drift-free guard that makes the graph the ONLY thing the CEO can do.
 *
 * The spine owns "what is next", gate routing, iteration caps, fork/join
 * barriers, director expansion, and graph edits — so the CEO never chooses
 * the next step and cannot drift. On top of that, at `agent/created` the
 * plugin denies the CEO's coding/external tools (bash, write, edit, web,
 * skill, …) for the CEO's own scope: the CEO is left with exactly the `wf_*`
 * tools, delegation, ask-user, and read-only tools. Node work therefore
 * happens inside the subagents the CEO spawns, which join the full composition
 * and are out of this guard's reach: their tools are what the delegation row that
 * spawns them allows. Removing the coding tools from the CEO is what turns "please
 * follow the graph" into "there is no other path". The CEO is every top-level
 * agent the plugin's tools reach: the agents of its composition when it is
 * mounted in one (a preset), every top-level agent on the host when it is
 * mounted at host level (the CLI graph profile).
 * @module @deepseek-ai/dsh-workflow-spine
 */

import type { Context } from '@deepseek-ai/cordis'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { setTimeout as wait } from 'node:timers/promises'
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, statSync, lstatSync, unlinkSync, writeFileSync } from 'node:fs'
import { basename, dirname, join, resolve, sep } from 'node:path'
import { homedir } from 'node:os'
import z from '@deepseek-ai/schemastery'
import { scopeChainOf, scopeOf } from '@deepseek-ai/dsh-scope'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { UserQuestionService } from '@deepseek-ai/dsh-user-questions'
import type { JsonValue } from '@deepseek-ai/dsh-util-values'
import { HALT, Spine, validateGraph } from './spine.ts'
import type { NodeView, SpineNext } from './spine.ts'
import { commandOf, isDeterministic, runDataCommand, runDeterministic, runWarningsOf } from './execute.ts'
import type { CommandFailure, CommandOutcome } from './execute.ts'
import type { WorkflowGraph } from './types.ts'
import { validateBuild } from './build-stamp.ts'
import { validateWorkspaceBuild } from './workspace-stamp.ts'
import { presentCall, presentResult, presentationMeta } from './present.ts'

export { Spine, validateGraph } from './spine.ts'
export { commandOf, dataCommandOf, isDeterministic, reasonOf, runDataCommand, runWarningsOf, verdictOf, NODE_ENV_KEY, REASON_LIMIT } from './execute.ts'
export type { CommandFailure, CommandOutcome, CommandRunnerConfig, NodeManifest } from './execute.ts'
export type { DataPass, GraphIssue, NodeView, PromptAmendmentMode, SpineNext } from './spine.ts'
export type { AgentWorkspace, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types.ts'
export { BUILD_STAMP, computeSourceStamp, validateBuild } from './build-stamp.ts'

export const name = 'workflow-spine'
export const inject = ['tools', 'systemPrompt']

/** Configured-only driver: the same drain/status path used by `wf_next`. */
export const CONFIGURED_RUN_SERVICE = 'workflowConfiguredRun'

/** Trusted one-shot execution, without a model interpreting user prose. */
export interface ConfiguredRun {
  /** Every declared node is runtime-owned, including every data/fork branch. */
  readonly automatic: boolean
  /** Advance the configured spine; unfinished, paused and failed states remain visible. */
  advance(): Promise<JsonValue>
}

/** Config: the workflow graph, tool naming, iteration bounds, and the CEO deny-list. */
export interface Config {
  /**
   * The workflow graph (refactory's `Workflow.to_dict()` output). Required when
   * {@link interactiveOpen} is false: the composition's one graph, validated at
   * mount. Omitted when {@link interactiveOpen} is true, where a graph arrives
   * later through `<toolPrefix>_open`.
   */
  graph?: WorkflowGraph
  /** Tool-name prefix (default `wf`, yielding `wf_next`, `wf_done`, …). */
  toolPrefix: string
  /** Maximum reloop iterations before a gate halts. */
  maxIterations: number
  /**
   * Open a mode's graph after boot, owned by the session that opens it, instead
   * of configuring one graph at mount.
   *
   * This is what a long-lived web chat session needs: it is one process serving
   * many sessions, so the graph cannot be a mount-time constant. When true the
   * spine starts with no graph, `graph` may be omitted, nothing is validated at
   * mount, and `<toolPrefix>_open` is registered for a session to open and claim
   * a run. When false (the default) the composition behaves exactly as before:
   * `graph` is required, validated at mount, and one spine serves the process.
   */
  interactiveOpen: boolean
  /**
   * Whether a person is attached to this composition. True when the graph was
   * started with `--chat` (terminal) or `interactiveOpen` (GUI CEO). A headless
   * CEO with `personAttached === false` cannot write `source='user-relayed'` via
   * `wf_steer`; the runtime silently downgrades to `'ceo'`.
   */
  personAttached?: boolean
  /**
   * The command that lowers a mode and prints its JSON payload, called as
   * `<openCommand> <openArgs...> <mode>` (with `--workdir <dir>` and one
   * `--knob NAME=VALUE` per knob appended). The payload names the run directory
   * (`run`), the graph name (`name`), the absolute path of the lowered graph
   * JSON (`graph`), and the run's declarations (`declarations`). Used by
   * `<toolPrefix>_open`; the console's `open` verb is that command.
   */
  openCommand: string
  /** Arguments for {@link openCommand}, before the mode. */
  openArgs: string[]
  /**
   * Run deterministic nodes (`FnNode`/`Study` commands and `fn` gate evaluators)
   * in-process instead of handing them to the driving agent. Requires a shell
   * executor; with none mounted the nodes are handed over as before.
   */
  executeCommands: boolean
  /** Working directory for node commands; absent uses the shell executor's own. */
  workdir?: string
  /**
   * Fallback environment entries for node commands.
   *
   * The values are per **run** — the work directory and the mode's knobs — so a
   * run opened at run time carries its own in the `<toolPrefix>_open` payload and
   * this is only what it falls back to when the payload omits one. A configured
   * graph has no payload, so this is the whole environment there and that path is
   * unchanged.
   */
  commandEnv: Record<string, string>
  /**
   * The command that selects a data node's items and opens and closes one pass,
   * run as `<dataCommand> <list|begin|end> --node <id> [--index <n>]`;
   * `reset` accepts a redo plan on JSON stdin to invalidate current facts. Required
   * when the graph has a `DataNode` and `executeCommands` is on: the items live
   * outside the graph, so a composition with no loader cannot iterate one. A run
   * opened at run time may name its own in the `<toolPrefix>_open` payload; this
   * is the fallback when it does not.
   */
  dataCommand: string
  /** Optional shared agent artifact/provenance bridge, using a JSON stdin protocol. */
  agentCommand?: string
  /** Optional review-scoped user-gate ledger bridge, using a JSON stdin protocol. */
  gateCommand?: string
  /**
   * The command that reads the run's steering ledger, run as
   * `<steerCommand> <steerArgs...> <workdir> [--note=<text> --kind=<kind>]`.
   * When non-empty (and `workdir` is set) the plugin registers `<toolPrefix>_steer`, so a person's mid-run
   * correction reaches the driving agent. The command's owner is the ledger
   * format's owner: the spine learns only a command to run and returns the JSON
   * it prints, so it stays free of any knowledge of how a ledger is stored.
   */
  steerCommand: string
  /** Arguments for {@link steerCommand}; the workdir and optional note flags follow. */
  steerArgs: string[]
  /** Foreground timeout for one node command. */
  commandTimeoutMs: number
  /**
   * How many node commands one dispatch boundary may hold in flight at once.
   * A fork's peers are concurrent by the graph's own declaration, but the
   * resources they share — one model gateway, one host — are not free: this is
   * the composition's ceiling on that fan, with peers beyond it queued in cursor
   * order until a slot frees. Zero (the default) is unbounded, which keeps every
   * peer of a boundary running at once.
   */
  maxConcurrentNodes?: number
  /** Sandbox mode for node commands; graph commands are trusted ops, so full access. */
  commandSandboxMode: string
  /**
   * Global tools the CEO is not allowed to call. The guard denies these on the
   * CEO's own scope at `agent/created`: every top-level agent of the mount's
   * composition, or of the whole host when the plugin is mounted at host level.
   * Subagents (which join the full composition as siblings) are out of its
   * reach; their delegation row's `toolFilter` decides what they keep. Names
   * absent from the CEO's inherited surface are skipped, so the list can name
   * platform-gated tools (e.g. `pwsh`) that are not present.
   */
  ceoDeny: string[]
}

/** Coding/external tools the CEO is not allowed to call, by default. */
const DEFAULT_CEO_DENY = [
  'bash',
  'pwsh',
  'write',
  'edit',
  'str_replace_editor',
  'web_search',
  'web_fetch',
  'skill',
  'ralph',
  'workflow',
  'read_image',
]

export const Config = z.object({
  graph: z.any(),
  toolPrefix: z.string().default('wf'),
  maxIterations: z.natural().default(3),
  interactiveOpen: z.boolean().default(false),
  personAttached: z.boolean().default(false),
  openCommand: z.string().default(''),
  openArgs: z.array(z.string()).default([]),
  executeCommands: z.boolean().default(false),
  workdir: z.string(),
  commandEnv: z.dict(z.string()).default({}),
  dataCommand: z.string().default(''),
  agentCommand: z.string().default(''),
  gateCommand: z.string().default(''),
  steerCommand: z.string().default(''),
  steerArgs: z.array(z.string()).default([]),
  commandTimeoutMs: z.natural().default(600_000),
  maxConcurrentNodes: z.natural().default(0),
  commandSandboxMode: z.string().default('danger-full-access'),
  ceoDeny: z.array(z.string()).default(DEFAULT_CEO_DENY),
})

/** How much of a failed node's stderr the status file keeps. */
const STATUS_DETAIL = 2_000

/**
 * Extract the meaningful failure detail from a node command's stderr.
 *
 * When the runner's `--- refactory failure ---` marker is present, take
 * only the text after the last occurrence (the structured block
 * `format_failure_detail` printed).  Otherwise fall back to head + tail,
 * never tail-only (dm-203).
 */
function failureDetailOf(stderr: string): string {
  const marker = '--- refactory failure ---'
  const markerIndex = stderr.lastIndexOf(marker)
  if (markerIndex !== -1) {
    const afterMarker = stderr.slice(markerIndex + marker.length).trimStart()
    return afterMarker.slice(0, STATUS_DETAIL)
  }
  // Fallback for non-refactory commands: head + tail, never tail-only.
  if (stderr.length <= STATUS_DETAIL) return stderr
  const head = stderr.slice(0, 500)
  const tail = stderr.slice(-(STATUS_DETAIL - 500 - 5))
  return `${head}\n…\n${tail}`
}

/**
 * What asking a person at a user gate came to: a verdict with any feedback; text
 * that is no answer, which leaves the gate due for the CEO to reply to; or nothing,
 * when the run was halted, paused or changed while the question was open.
 */
type UserGateResult =
  | { verdict: string; feedback: string }
  | { status: 'replied'; gate: string; text: string }
  | undefined

/**
 * Read a typed gate answer. An exact outcome, or `outcome: feedback`, is a verdict;
 * anything else is conversation for the CEO, so nothing is guessed on the person's behalf.
 * @param text - what the person typed.
 * @param labels - the outcomes the gate offers.
 * @returns the verdict and its feedback, or undefined when the text is no outcome.
 */
function readGateAnswer(text: string, labels: string[]): { verdict: string; feedback: string } | undefined {
  if (labels.length === 0) return { verdict: text, feedback: '' }
  const lower = text.toLowerCase()
  const exact = labels.find(label => label.toLowerCase() === lower)
  if (exact !== undefined) return { verdict: exact, feedback: '' }
  // Longest label first, so one label that prefixes another cannot claim its answers.
  const prefixed = [...labels].sort((a, b) => b.length - a.length)
    .find(label => lower.startsWith(label.toLowerCase() + ':'))
  return prefixed === undefined ? undefined : { verdict: prefixed, feedback: text.slice(prefixed.length + 1).trim() }
}

/** One model-visible step, rendered as compact JSON text. */
function text(value: JsonValue): { type: 'text'; text: string } {
  return { type: 'text', text: JSON.stringify(value, null, 2) }
}

/** Cast a spine result to the JSON the tool boundary requires. */
function json(value: unknown): JsonValue {
  return value as JsonValue
}

/** Wrap a synchronous spine result in the Promise the tool contract requires. */
function sync<T>(value: T): Promise<T> {
  return Promise.resolve(value)
}

/** Map a node's _type to a semantic kind a reader can understand. */
function nodeKind(_type: string): 'agent' | 'operation' | 'gate' | 'control' {
  switch (_type) {
    case 'AgentNode':
    case 'LLMNode':
      return 'agent'
    case 'GateNode':
      return 'gate'
    case 'ForkNode':
    case 'JoinNode':
    case 'DataNode':
      return 'control'
    case 'FnNode':
    case 'Study':
    case 'SelectionNode':
      return 'operation'
    default:
      return 'operation'
  }
}

/** The last non-empty line of a command's output, for an error a reader can act on. */
function lastLine(text: string | null | undefined): string {
  const lines = (text ?? '').split('\n').map(line => line.trim()).filter(line => line !== '')
  return lines[lines.length - 1] ?? ''
}

/**
 * What one deterministic node left in its run directory, for the CEO to report.
 *
 * A node's outputs are the files it declared it writes and that now exist; a
 * declaration that names a file the command did not produce is not an output,
 * and saying otherwise would let the CEO report work that did not happen. The
 * facts are the run's own `facts.json`, which the node command rewrites and
 * which is empty when no command has run yet.
 * @param node - the node that just ran.
 * @param workdir - the run directory its command ran in.
 * @param peers - every node this dispatch boundary ran, when the cursor held
 * more than one; the boundary's last peer in cursor order is the named node.
 * @param outcome - what the named node's command returned, so a gate's verdict
 * and reason are reported with it rather than left for the CEO to infer.
 * @returns the node's id and type, any verdict and reason, plus the outputs and
 * facts that exist.
 */
function recordedStep(
  node: NodeView,
  workdir: string | undefined,
  peers?: NodeView[],
  outcome?: CommandOutcome,
): RanStep {
  const step: RanStep = { node: node.id, type: node._type }
  if (peers !== undefined && peers.length > 1) step.peers = peers.map(peer => peer.id)
  if (outcome?.ok === true && outcome.verdict !== undefined) {
    step.verdict = outcome.verdict
    if (outcome.reason !== undefined) step.reason = outcome.reason
  }
  if (workdir === undefined) return step
  const outputs = (node.writes ?? []).filter(rel => rel !== '' && existsSync(resolve(workdir, rel)))
  if (outputs.length > 0) step.outputs = outputs
  try {
    // The runner writes one JSON object; anything else is not facts a reader can
    // use, and is reported as absent rather than as a malformed value.
    const parsed: unknown = JSON.parse(readFileSync(join(workdir, 'facts.json'), 'utf8'))
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      // E10: 512 characters keeps short strings, numbers, booleans, and small
      // arrays visible in the CEO's wf_next result while excluding large blobs
      // like file contents that would fill the context window (a design run's
      // facts.json is about 37 KB).  Values whose JSON serialization exceeds
      // this limit are reported by key and size only; the full value is not
      // available through wf_next (the CEO's coding tools are disabled while
      // driving a graph).
      const FACT_INLINE_LIMIT = 512
      const factKeys = Object.keys(parsed as Record<string, unknown>)
      const sizes: Record<string, number> = {}
      const inline: Record<string, unknown> = {}
      for (const key of factKeys) {
        const size = JSON.stringify((parsed as Record<string, unknown>)[key]).length
        sizes[key] = size
        if (size <= FACT_INLINE_LIMIT) inline[key] = (parsed as Record<string, unknown>)[key]
      }
      step.facts = { _summary: true, path: 'facts.json', keys: factKeys, sizes, inline } as unknown as JsonValue
    }
  } catch {
    // No facts file yet, or one a command left mid-write: the run directory
    // simply carries no facts for this step.
  }
  return step
}

/**
 * A gate prompt as the question card puts it: its last paragraph is the ask, the title
 * the card shows on one line, and the paragraphs before it are the detail the card
 * renders as text (dm-163). A prompt of one paragraph has no detail.
 * @param prompt - the gate's prompt, its facts filled.
 * @returns the ask, and the detail when there is one.
 */
export function splitGatePrompt(prompt: string): { ask: string; detail?: string } {
  const paragraphs = prompt.trim().split(/\n[ \t]*\n/u)
  const ask = paragraphs.pop()!.trim()
  const detail = paragraphs.join('\n\n').trim()
  return detail === '' ? { ask } : { ask, detail }
}

/**
 * A gate's prompt with the run's facts filled in, as refactory fills `{{name}}` in an
 * agent node's prompt (`render`): a fact the run has not produced is left as written.
 *
 * A user gate's question is all its card shows, so a gate that must say something only
 * the run knows -- design mode's names the repository delivery would create -- carries
 * it as a fact, and it is filled wherever the question is put.
 * @param text - the gate's prompt, as lowered.
 * @param workdir - the run directory whose `facts.json` holds the values.
 * @returns the prompt with every known `{{name}}` replaced.
 */
export function withFacts(text: string, workdir: string | undefined): string {
  if (workdir === undefined || !text.includes('{{')) return text
  let facts: Record<string, unknown>
  try {
    const parsed: unknown = JSON.parse(readFileSync(join(workdir, 'facts.json'), 'utf8'))
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return text
    facts = parsed as Record<string, unknown>
  } catch {
    return text
  }
  return text.replace(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g, (whole: string, key: string) => {
    if (!Object.hasOwn(facts, key)) return whole
    const value = facts[key]
    return typeof value === 'string' ? value : JSON.stringify(value)
  })
}

/**
 * The run directory a session gets when the person names none.
 *
 * A chat has no terminal to name a directory, and the console's `open` verb
 * refuses without one, so the spine derives a directory from the two facts it
 * has at the call: the calling session and the mode. It is
 * `<process.cwd()>/.factory/web/<session id>/<package>-<mode>`, where the mode's
 * `/` becomes `-` so the leaf reads `<package>-<mode>`, and it is created here
 * because the open command will lower into it.
 * @param session - the id of the session opening the run.
 * @param mode - the mode, as `<package>/<name>`.
 * @returns the absolute run directory, created.
 */
function derivedWorkdir(session: string, mode: string): string {
  const base = join(process.cwd(), '.factory', 'web', session, mode.split('/').join('-'))
  let dir = base
  for (let suffix = 2; existsSync(dir); suffix += 1) dir = `${base}-${suffix}`
  mkdirSync(dir, { recursive: true })
  return dir
}

/** Resolve existing symlink ancestors even when the requested leaf is new. */
function canonicalWorkdir(path: string): string {
  let ancestor = resolve(path.replace(/^~(?=[/\\]|$)/, homedir()))
  const tail: string[] = []
  while (!existsSync(ancestor)) {
    tail.unshift(basename(ancestor))
    ancestor = dirname(ancestor)
  }
  return resolve(realpathSync(ancestor), ...tail)
}

/** The per-call execution facts a spine tool resolves its session from. */
interface ToolExec {
  agent?: Agent
  signal?: AbortSignal
}

/**
 * One dispatch boundary a `wf_next` (or `wf_open`) ran for the CEO.
 *
 * With `interactiveOpen` a call runs exactly one dispatch boundary — every
 * deterministic node the cursor holds at once, dispatched together as fork
 * peers (#306) — so the result can name a node that just ran and what it left
 * in the run directory alongside the nodes still due; `peers` names the whole
 * boundary when more than one ran. Neither field is invented when the run
 * directory does not carry it: a node that declares no artifact has no
 * `outputs`, and a run whose commands have not written `facts.json` has no
 * `facts`.
 */
interface RanStep {
  /** A node that just ran: the boundary's last peer in cursor order. */
  node: string
  /** Its node type, so a reader can tell an op from a gate. */
  type: string
  /** Every node this boundary dispatched, when the cursor held more than one. */
  peers?: string[]
  /** The verdict the node routed on, when it is a gate the runtime evaluated. */
  verdict?: string
  /** Why it routed that way: what its evaluator printed before the verdict. */
  reason?: string
  /** The node's declared writes that now exist in the run directory, relative to it. */
  outputs?: string[]
  /** The run's `facts.json` as the node's command left it. */
  facts?: JsonValue
  /** The run warnings the boundary's nodes printed on stderr, each led by its node's id. */
  warnings?: string[]
}

/** The item a data pass already materialised on disk, so it is begun once per item. */
interface Materialised {
  /** `<node>:<index>` of the pass whose item is on disk. */
  key: string
  /** The item `begin` printed for that pass. */
  item: unknown
}

/** A graph opened at run time, and the session that owns it. */
interface OpenRun {
  /** The id of the session that opened the run. */
  session: string
  /** The opened graph's spine. */
  spine: Spine
  /** The run directory the mode was lowered into: where node commands run and status is written. */
  workdir: string
  /** The opened graph's name, for status and reporting. */
  name: string
  /** The run's own environment for node commands, from the open payload. */
  commandEnv: Record<string, string>
  /** The run's own item loader command, from the open payload. */
  dataCommand: string
  /** Optional shared agent artifact/provenance bridge, using a JSON stdin protocol. */
  agentCommand: string
  /** Shared user-gate decision bridge. */
  gateCommand: string
}

/** The JSON `<openCommand>` prints: where it lowered the mode. */
interface OpenPayload {
  /** The run directory the mode was lowered into. */
  run?: unknown
  /** The lowered graph's name. */
  name?: unknown
  /** Absolute path of the lowered graph JSON the spine loads. */
  graph?: unknown
  /** Absolute path of the run's node declarations. */
  declarations?: unknown
  /**
   * The run's own environment for node commands: the work directory, the project,
   * the knobs and the interpreter-leading `PATH`. Per run, so it travels with the
   * payload rather than being a mount-time constant.
   */
  commandEnv?: unknown
  /** The run's own data-node item loader command. */
  dataCommand?: unknown
  /** Optional agent artifact/provenance bridge owned by the opened mode runtime. */
  agentCommand?: unknown
  /** Shared user-gate decision bridge. */
  gateCommand?: unknown
}

/** The environment a payload names, or `undefined` when it is absent or malformed. */
function payloadEnv(value: unknown): Record<string, string> | undefined {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return undefined
  const out: Record<string, string> = {}
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    // A non-string value is not an environment a process can be given, so the
    // whole payload is treated as absent and the config fallback is used.
    if (typeof entry !== 'string') return undefined
    out[key] = entry
  }
  return out
}

/**
 * Stable stringify replacer: recursively sorts object keys for deterministic
 * JSON serialization. Arrays are preserved in order; primitives pass through.
 */
function stableStringify(_key: string, value: unknown): unknown {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0))
  }
  return value
}

/**
 * The graph revision a person's preview names and a rerun must still match (dm-252):
 * a SHA-256 of a checkpoint's graph-structural fields.
 *
 * Runtime metadata (owner_pid, status, file_fingerprints, materialised, run_id,
 * graph_revision itself) and ephemeral state (pause_before, halted, log,
 * skip_provenance, pending_skip_receipts) are excluded, so the revision is stable
 * across owner restarts, acknowledgements and pause/resume cycles.
 * @param saved - a checkpoint, as `Spine.checkpoint()` returns it or as it was saved.
 * @returns the hex digest.
 */
export function computeGraphRevision(saved: Record<string, unknown>): string {
  const state = saved.state as Record<string, unknown>
  const structural = {
    schema_version: saved.schema_version,
    source_graph: saved.source_graph,
    graph: saved.graph,
    afters_map: saved.afters_map,
    successors: saved.successors,
    redo_plans: saved.redo_plans,
    deferred_redo_plans: saved.deferred_redo_plans,
    cursor: state.cursor,
    completed: state.completed,
    skipped: state.skipped,
    gate_verdicts: state.gate_verdicts,
    iterations: state.iterations,
    feedback: state.feedback,
    gate_reasons: state.gate_reasons,
    prompts: state.prompts,
    pass: state.pass,
  }
  return createHash('sha256')
    .update(JSON.stringify(structural, stableStringify))
    .digest('hex')
}

/**
 * Build the argv for {@code wf_open}, including the session's project when set.
 *
 * Exported so the project-propagation tests call the real logic instead of
 * re-implementing it inline.
 */
export function buildWfOpenArgv(
  openArgs: readonly string[],
  sessionCwd: string | undefined,
  mode: string,
): string[] {
  const argv = [...openArgs]
  if (typeof sessionCwd === 'string' && sessionCwd !== '') {
    argv.push('--project', sessionCwd)
  }
  argv.push(mode)
  return argv
}

export function apply(ctx: Context, config: Config): void {
  // Detect stale builds early, before any graph is configured or opened.
  // A stale build is a developer-infrastructure error with no recovery path —
  // the engine and Python halves would disagree on node dispatch — so exit
  // immediately instead of throwing into Cordis (which catches, marks the
  // plugin inactive, and hangs the process waiting for services that never
  // arrive).
  try {
    validateBuild()
  } catch (err) {
    process.stderr.write(`\n${err instanceof Error ? err.message : String(err)}\n\n`)
    process.exit(1)
  }
  try {
    validateWorkspaceBuild()
  } catch (err) {
    // Workspace stamp mismatch is a warning, not fatal: a stale workspace build
    // must not kill a spine mount mid-run, during a resume, or at GUI boot.
    // The per-package validateBuild() above remains fatal.
    process.stderr.write(
      `\nWARNING: ${err instanceof Error ? err.message : String(err)}\n\n`,
    )
  }

  const { toolPrefix, maxIterations, ceoDeny, interactiveOpen } = config
  const effectivePersonAttached = interactiveOpen || config.personAttached
  // This is initialized before an explicit checkpoint restore at mount time.
  const materialised = new WeakMap<Spine, Materialised>()

  // The composition's one spine, when the graph was a mount-time constant. When
  // `interactiveOpen` is true there is none yet: a session opens a mode through
  // `<toolPrefix>_open`, and only then is a graph validated and a spine built.
  let configuredSpine: Spine | undefined
  if (!interactiveOpen) {
    if (config.graph === undefined) {
      throw new Error(
        'no workflow graph is configured; set `graph`, or set `interactiveOpen: true` so a '
        + `session can open one with ${toolPrefix}_open`,
      )
    }
    // A graph accepted from configuration is a parser boundary: reject it at load
    // rather than discovering a dead cursor mid-run.
    const issues = validateGraph(config.graph)
    if (issues.length > 0) {
      const rendered = issues
        .map(issue => (issue.node === undefined ? issue.message : `${issue.node}: ${issue.message}`))
        .join('; ')
      throw new Error(`invalid workflow graph: ${rendered}`)
    }
    // A data node's items live outside the graph, so a composition that mounts no
    // loader hands the node to the driving agent, which cannot iterate a subgraph the
    // cursor never enters. Refused here, at load, because both facts are config.
    const dataNodes = Object.values(config.graph.nodes)
      .filter(node => node._type === 'DataNode')
      .map(node => node.id)
    if (dataNodes.length > 0) {
      const reason = !config.executeCommands
        ? 'node commands are disabled (`executeCommands: false`)'
        : config.dataCommand === ''
          ? 'no item loader is configured (`dataCommand`)'
          : ''
      if (reason !== '') {
        throw new Error(
          `this graph iterates data in ${dataNodes.join(', ')}, and ${reason}; a data node's `
          + 'items are selected by a command outside the graph, so a composition without one '
          + 'cannot run it',
        )
      }
    }
    configuredSpine = new Spine(config.graph, { maxIterations })
    restoreRequested(configuredSpine, config.workdir)
  }

  // Tools are shared by the web composition; cursors and run settings are not.
  // Keep each claim until that same session successfully opens its next run.
  const openedBySession = new Map<string, OpenRun>()

  // Child-to-parent session map for the write-confinement listener (#347, C0).
  // Populated at `agent/created` when the created agent has a parentSession,
  // cleaned at `agent/disposed`. The map makes the parent-chain walk possible:
  // from a grandchild to the session that opened the run.
  const childToParent = new Map<string, string>()

  // Subagent write confinement (#347, C0). A `tools/pre-execute` listener that
  // denies `write` and `edit` calls whose resolved path falls outside the open
  // run's directory, when the calling session is a delegated child of a session
  // that has an open run.
  //
  // Path resolution mirrors dsh-tool-fs: the sandbox policy's workspaceRoot
  // (when a sandboxPolicy service is mounted) takes precedence over
  // session.header.cwd, which is how the write/edit tools themselves resolve
  // file_path. Both the run directory and the target are normalised with
  // canonicalWorkdir (realpathSync for existing ancestors, resolve for new
  // segments), so /var vs /private/var discrepancies are handled.
  //
  // The first step up the chain comes from the calling session's own
  // header.parentSession, so a direct child is confined even if its
  // agent/created event was not observed. Subsequent steps use the
  // childToParent map.
  ctx.on('tools/pre-execute', (exec, next) => {
    // 1. Only intercept 'write' and 'edit' tool calls
    if (exec.name !== 'write' && exec.name !== 'edit') return next()

    // 2. Get the calling session
    const session = exec.agent?.session
    if (session === undefined) return next()

    // 3. Identify a delegated child by structural lineage
    const depth = session.header.delegationDepth
    const parent = session.header.parentSession
    if ((depth === undefined || depth === 0) && parent === undefined) return next()

    // 4. Walk the parent chain to find the run directory.
    //    First step: the session's own header.parentSession (so a direct child
    //    is confined even if its agent/created event was not seen).
    //    Subsequent steps: the childToParent map.
    let runDir: string | undefined
    let current: string | undefined = parent !== undefined ? String(parent) : undefined

    for (let steps = 0; steps < 20 && current !== undefined; steps += 1) {
      const opened = openedBySession.get(current)
      if (opened !== undefined) {
        runDir = opened.workdir
        break
      }
      current = childToParent.get(current)
    }

    // 5. Fallback to config.workdir for configured-graph mode
    if (runDir === undefined && configuredSpine !== undefined) {
      runDir = config.workdir
    }

    // 6. No confinement boundary found — allow freely
    if (runDir === undefined) return next()

    // 7. Resolve the target file_path exactly as dsh-tool-fs does:
    //    sandbox policy's workspaceRoot when available, else session.header.cwd.
    const filePath = (exec.arguments as Record<string, unknown>)?.file_path
    if (typeof filePath !== 'string') return next()
    const sandboxPolicyService: { resolve(opts: { session?: unknown }): { workspaceRoot: string } } | undefined =
      ctx.get('sandboxPolicy')
    const policyCwd = sandboxPolicyService !== undefined
      ? sandboxPolicyService.resolve({ session }).workspaceRoot
      : undefined
    const sessionCwd = policyCwd ?? session.header.cwd
    const resolvedTarget = canonicalWorkdir(
      sessionCwd !== undefined ? resolve(sessionCwd, filePath) : resolve(filePath),
    )

    // 8. Normalise the run directory the same way
    const resolvedRun = canonicalWorkdir(runDir)

    // 9. Compare: target must be inside or equal to the run directory
    if (resolvedTarget !== resolvedRun
        && !resolvedTarget.startsWith(resolvedRun.endsWith(sep) ? resolvedRun : resolvedRun + sep)) {
      return Promise.resolve({
        kind: 'deny' as const,
        reason: `this subagent may only write inside the run directory ${resolvedRun}`,
      })
    }

    // 10. Deny writes to .refactory/ — spine state must not be forged by subagents (#347, C1).
    // Compare case-insensitively: on macOS's default filesystem, .REFACTORY
    // resolves to .refactory, so a case-folded prefix check closes the bypass (#347, C1).
    const refactoryDir = join(resolvedRun, '.refactory')
    const targetLower = resolvedTarget.toLowerCase()
    const refactoryLower = refactoryDir.toLowerCase()
    if (targetLower === refactoryLower
        || targetLower.startsWith(refactoryLower + sep.toLowerCase())) {
      return Promise.resolve({
        kind: 'deny' as const,
        reason: `this subagent may not write inside ${refactoryDir}`,
      })
    }

    return next()
  })

  // Completion override records: keyed by session id so parallel CEO sessions
  // do not share a single variable (#347, C1).  When the C1 pre-execute listener
  // allows `update_goal complete` via a person-confirmed override, it stores the
  // description here so the C1 post-execute listener can append it to the tool
  // result, recording the override on the goal's completion.
  const pendingCompletionOverrides = new Map<string, string>()

  // Goal completion guard (#347, C1). A `tools/pre-execute` listener that
  // denies `update_goal` with action `complete` when the run's recorded status
  // (spine-status.json) is not `done` and no person-issued `completion_override`
  // matches the current status. The refusal steers the CEO toward `update_goal
  // blocked` (from round 3) or ending the turn (rounds 1–2).
  ctx.on('tools/pre-execute', (exec, next) => {
    if (exec.name !== 'update_goal') return next()
    const args = exec.arguments as Record<string, unknown> | undefined
    if (args?.action !== 'complete') return next()

    // Find the run workdir the same way C0 does.
    let workdir: string | undefined
    if (configuredSpine !== undefined) {
      workdir = config.workdir
    } else {
      const session = exec.agent?.session
      if (session !== undefined) {
        const sid = String(session.id)
        const opened = openedBySession.get(sid)
        if (opened !== undefined) workdir = opened.workdir
        else {
          let current: string | undefined = session.header.parentSession !== undefined
            ? String(session.header.parentSession)
            : undefined
          for (let steps = 0; steps < 20 && current !== undefined; steps += 1) {
            const o = openedBySession.get(current)
            if (o !== undefined) { workdir = o.workdir; break }
            current = childToParent.get(current)
          }
        }
      }
    }

    const statusRecord = readStatusRecord(workdir)
    if (statusRecord === undefined) return next() // no status file yet — allow
    const currentStatus = typeof statusRecord.status === 'string' ? statusRecord.status : ''
    if (currentStatus === 'done') return next()

    // Check completion_override (status AND invocation must both match)
    const override = statusRecord.completion_override
    if (override !== null && typeof override === 'object' && !Array.isArray(override)) {
      const ov = override as Record<string, unknown>
      if (ov.confirmed_status === currentStatus) {
        // Validate invocation: the override is bound to the invocation that confirmed it,
        // so a resumed or new invocation does not inherit it (#347, C1).
        const overrideInvocation = ov.invocation ?? null
        let currentInvocation: unknown = null
        try {
          currentInvocation = JSON.parse(readFileSync(join(workdir!, '.refactory', 'invocation.json'), 'utf8')).id ?? null
        } catch { /* no invocation file — invocation is null */ }
        if (overrideInvocation === currentInvocation) {
          // Record the override so the post-execute listener injects it into
          // the goal's completion result ("the goal then records that").
          const overrideSessionId = exec.agent?.session ? String(exec.agent.session.id) : '_configured_'
          pendingCompletionOverrides.set(overrideSessionId,
            `[Completion allowed via person-confirmed override: `
            + `status="${currentStatus}", confirmed at ${String(ov.at ?? 'unknown')}]`)
          return next()
        }
      }
    }

    return Promise.resolve({
      kind: 'deny' as const,
      reason: `The run's recorded status is "${currentStatus}". `
        + 'End this turn and ask the person whether to continue or mark the goal blocked; '
        + 'from the third goal round you may use update_goal with action "blocked".',
    })
  })

  // Goal completion override recording (#347, C1). When the pre-execute
  // listener allowed `update_goal complete` via a person-confirmed override,
  // append the override note to the tool result so it is recorded on the goal.
  ctx.on('tools/post-execute', (exec, result, next) => {
    if (exec.name !== 'update_goal') return next()
    const sessionId = exec.agent?.session ? String(exec.agent.session.id) : '_configured_'
    const note = pendingCompletionOverrides.get(sessionId)
    if (note === undefined) return next()
    pendingCompletionOverrides.delete(sessionId)
    if (result.isError) return next()
    return Promise.resolve({
      kind: 'accept' as const,
      content: [...result.content, { type: 'text' as const, text: note }],
    })
  })

  // Cancelled approval rewrite (#347, C3). A `tools/post-execute` listener that
  // rewrites `ask_user_question` cancellation errors to an unambiguous refusal.
  // Matches error.info.code against ASK_CANCELLED, ASK_ABORTED, ASK_TIMED_OUT.
  const ASK_CANCEL_CODES = new Set(['ASK_CANCELLED', 'ASK_ABORTED', 'ASK_TIMED_OUT'])
  ctx.on('tools/post-execute', (exec, result, next) => {
    if (exec.name !== 'ask_user_question') return next()
    if (!result.isError) return next()
    const code = result.error?.info?.code
    if (code === undefined || !ASK_CANCEL_CODES.has(code)) return next()
    return Promise.resolve({
      kind: 'block' as const,
      feedback: [{ type: 'text' as const, text: 'Not approved: stop and wait for the person.' }],
    })
  })

  // CEO prompt recording (#347, C2). Records the prompt text the CEO sends
  // through subagent, subagent_fork and send_message, keyed by runKey:node.
  // Used by wf_done to validate that review/audit nodes received the
  // pending delivery's prompt.txt.  Cleared on delivery completion and
  // redispatch so an earlier faithful prompt cannot satisfy a rewritten retry.
  const ceoPrompts = new Map<string, Array<{ prompt: string; hash: string }>>()

  /** Extract the node id from the execution contract embedded in a prompt. */
  function extractContractNode(text: string): string | undefined {
    const startMarker = '<!-- refactory worker execution -->'
    const endMarker = '<!-- end refactory worker execution -->'
    const start = text.indexOf(startMarker)
    if (start < 0) return undefined
    const end = text.indexOf(endMarker, start)
    if (end < 0) return undefined
    const block = text.substring(start + startMarker.length, end)
    const match = block.match(/"node"\s*:\s*"([^"]+)"/)
    return match ? match[1] : undefined
  }

  ctx.on('tools/pre-execute', (exec, next) => {
    let promptText: string | undefined
    if (exec.name === 'subagent' || exec.name === 'subagent_fork') {
      const args = exec.arguments as Record<string, unknown> | undefined
      promptText = typeof args?.prompt === 'string' ? args.prompt : undefined
    } else if (exec.name === 'send_message') {
      const args = exec.arguments as Record<string, unknown> | undefined
      promptText = typeof args?.message === 'string' ? args.message : undefined
    }
    if (promptText === undefined) return next()

    // Record only top-level CEO calls, not reviewer subagent calls (#347, C2).
    // A delegated child (depth > 0 or has a parentSession) is a subagent whose
    // send_message the tool filter allows but should not be recorded as a CEO prompt.
    const session = exec.agent?.session
    if (session !== undefined) {
      const depth = session.header.delegationDepth
      const parent = session.header.parentSession
      if ((depth !== undefined && depth > 0) || parent !== undefined) return next()
    }

    // Find due nodes from the spine, keyed by run (workdir) so parallel
    // GUI sessions cannot cross-contaminate (#347, C2).
    let spine: Spine | undefined
    let runKey: string = '_'
    if (configuredSpine !== undefined) {
      spine = configuredSpine
      runKey = config.workdir ?? '_configured_'
    } else {
      if (session !== undefined) {
        const opened = openedBySession.get(String(session.id))
        if (opened !== undefined) {
          spine = opened.spine
          runKey = opened.workdir
        }
      }
    }
    if (spine !== undefined) {
      const cur = spine.next()
      if (cur.status === 'ready') {
        const hash = createHash('sha256').update(promptText).digest('hex')
        // Attribute the prompt to a due node by its delivery key: the node id
        // from the execution contract embedded in prompt.txt (#347, C2).
        const contractNodeId = extractContractNode(promptText)
        if (contractNodeId !== undefined) {
          // The prompt names a specific node via its delivery key.
          const dueNode = cur.nodes.find(n => n.id === contractNodeId
            && (n._type === 'AgentNode' || n._type === 'LLMNode'))
          if (dueNode !== undefined) {
            // Record under this node — whether faithful or rewritten, wf_done
            // will check the latest recorded prompt for content matching.
            const key = `${runKey}:${dueNode.id}`
            const existing = ceoPrompts.get(key)
            if (existing !== undefined) existing.push({ prompt: promptText, hash })
            else ceoPrompts.set(key, [{ prompt: promptText, hash }])
          }
          // A contract naming a non-due node: record under none, clear nothing.
        }
        // A prompt with no contract (e.g. an unrelated send_message to a
        // continuable subagent) is recorded under none and no records are
        // cleared, so faithful reviewers are not refused (#347, C2).
      }
    }
    return next()
  })

  // Reviewer node ids, fetched once from the runtime bridge and cached.
  // undefined = not yet fetched; null = fetch failed, treat all nodes as reviewers (fail closed).
  let reviewerIds: Set<string> | null | undefined
  // Pending delivery prompts, cached per runKey:node at begin time.
  // Keyed by `${runKey}:${nodeId}` so parallel GUI sessions do not share state.
  const pendingPrompts = new Map<string, string>()
  // Unresolved template keys per runKey:node, cached at begin for inclusion in receipt (#347, C2).
  const pendingUnresolvedKeys = new Map<string, unknown[]>()
  // Track which deliveries have already warned about unresolved keys, so we warn
  // once per delivery, not per wf_next call (#347, C2).
  const unresolvedKeyWarned = new Set<string>()

  /** The id of the session an execution belongs to, when it has an agent. */
  function sessionIdOf(exec: ToolExec): string | undefined {
    const session = exec.agent?.session
    return session === undefined ? undefined : String(session.id)
  }

  /**
   * The spine a tool call works on, and the per-run settings it belongs to.
   *
   * With a configured graph that is the composition's one spine and its static
   * configuration. With `interactiveOpen` it is the run the calling session
   * opened — its own work directory, environment and item loader, read from the
   * open payload — and a call from any other session, or before any open, is
   * refused rather than served against an unknown run.
   */
  function requireRun(exec: ToolExec): {
    spine: Spine
    workdir: string | undefined
    commandEnv: Record<string, string>
    dataCommand: string
    agentCommand: string
    gateCommand: string
  } {
    if (configuredSpine !== undefined) {
      return {
        spine: configuredSpine,
        workdir: config.workdir,
        commandEnv: config.commandEnv,
        dataCommand: config.dataCommand,
        agentCommand: config.agentCommand ?? '',
        gateCommand: config.gateCommand ?? '',
      }
    }
    const owner = sessionIdOf(exec)
    const opened = owner === undefined ? undefined : openedBySession.get(owner)
    if (opened === undefined) {
      throw new Error(`no mode is open in this session; call ${toolPrefix}_open with a mode first`)
    }
    return {
      spine: opened.spine,
      workdir: opened.workdir,
      commandEnv: opened.commandEnv,
      dataCommand: opened.dataCommand,
      agentCommand: opened.agentCommand,
      gateCommand: opened.gateCommand,
    }
  }

  // Who answers a `user` gate. Asked of the composition when a gate is asked, as
  // `shell` and `sandboxPolicy` are, never captured at apply: the loader applies
  // rows as their modules arrive, so the spine can be applied before the
  // user-questions row it sits beside, and a value read here stayed undefined for the
  // whole run, refusing every user gate at the terminal (dm-255). Routing one stays a
  // runtime act rather than a claim the model makes: a composition without an
  // answerer (a headless run) cannot route a person's gate at all, and a model cannot
  // route one on their behalf.
  const userQuestionsService = (): UserQuestionService | undefined => ctx.get('userQuestions')

  /** The one real way to answer a user gate the runtime could not put to a person. */
  function answerElsewhere(gate: string, labels: readonly string[], workdir: string | undefined): string {
    const run = workdir ?? '<run directory>'
    const label = labels.length === 0 ? '<label>' : `<${labels.join('|')}>`
    return `answer it with \`refactory steer --run ${run} --verdict ${gate}=${label} [--note "…"]\`, `
      + `then \`refactory graph-control resume --run ${run}\``
  }

  /** Call a runtime-owned bookkeeping bridge without editing artifact text. */
  function runtimeBridge(command: string, workdir: string, env: Record<string, string>, payload: unknown): Record<string, unknown> {
    const result = spawnSync(command, {
      shell: true, cwd: workdir, env: { ...process.env, ...env },
      input: JSON.stringify(payload), encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
      timeout: config.commandTimeoutMs,
    })
    if (result.status !== 0) throw new Error(lastLine(result.stderr) || 'runtime bookkeeping bridge failed')
    const value: unknown = JSON.parse(result.stdout)
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('runtime bookkeeping bridge returned no receipt')
    }
    return value as Record<string, unknown>
  }

  function restoreRequested(spine: Spine, workdir: string | undefined): void {
    if (workdir === undefined) return
    const requestPath = join(workdir, '.refactory', 'resume-request.json')
    if (!existsSync(requestPath)) return
    const request = JSON.parse(readFileSync(requestPath, 'utf8'))
    const bytes = readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'))
    if (createHash('sha256').update(bytes).digest('hex') !== request.checkpoint_sha256) throw new Error('checkpoint changed after recovery was requested')
    const saved = JSON.parse(bytes.toString())
    if (saved.status === 'executing') {
      if (request.acknowledge_executing !== true) throw new Error('unknown side-effect outcome; automatic replay refused')
      // With acknowledge_executing, treat as halted for restore purposes.
      // The checkpoint stays as-is; the spine sees an executing checkpoint
      // accepted only because the person inspected the side effects.
      saved.state.halted = undefined
      saved.state.pause_before = []
    }
    if (saved.status === 'failed' && request.acknowledge_failed !== true) throw new Error('last node outcome failed; pass acknowledge_failed to resume from this state')
    if (saved.state.halted?.reason === 'cancelled') delete saved.state.halted
    saved.state.pause_before = []
    spine.restore(saved)
    if (spine.passOf() !== undefined) {
      if (!Object.hasOwn(saved, 'materialised')) throw new Error('active data pass has no saved setup boundary; automatic replay refused')
      if (saved.materialised !== null) {
        const pass = spine.passOf()!
        if (saved.materialised?.key !== `${pass.node}:${pass.index}`) throw new Error('saved item setup belongs to another data pass')
        materialised.set(spine, saved.materialised)
      }
    }
    unlinkSync(requestPath)
  }

  function fingerprint(path: string): unknown {
    if (!existsSync(path)) return null
    if (!statSync(path).isDirectory()) return createHash('sha256').update(readFileSync(path)).digest('hex')
    const found: Record<string, string> = {}
    const visit = (directory: string, prefix: string): void => {
      for (const entry of readdirSync(directory).sort()) {
        const child = join(directory, entry), key = prefix + entry
        if (lstatSync(child).isSymbolicLink()) continue
        if (statSync(child).isDirectory()) visit(child, key + '/')
        else found[key] = createHash('sha256').update(readFileSync(child)).digest('hex')
      }
    }
    visit(path, '')
    return found
  }

  /**
   * The cursor's own status on disk, when a work directory is known.
   *
   * The runner cannot see the spine, so a run that stopped at a gate it could not
   * route used to look like a run that simply scored 0. The status file is what makes
   * "the graph finished" and "the graph refused" different facts to a reader.
   * @param spine - the run's spine, whose cursor the status records.
   * @param workdir - where to write: the configured run directory, or the opened
   * run's own directory when a session opened the mode at run time.
   * @param status - `done`, `ready`, `failed`, or `refused`.
   * @param given - the error, for a failure or refusal. A halt at a gate's reloop limit
   * without one records what the limit stopped and how to go on.
   * @param failedNode - the node that failed. Writes `last_failure` in the
   * status record. For most node command failures (nonzero exit other than 3)
   * the run-level status stays non-terminal (`ready`) so the retry is not
   * mistaken for a finished, failed run (dm-95). For a build-loop limit stop
   * (exit code 3 with the `--- refactory failure ---` marker), the run-level
   * status is `failed` — `last_failure` accompanies the terminal status so
   * readers see which node stopped the run.
   */
  function recordStatus(spine: Spine, workdir: string | undefined, status: string, given?: string,
      failedNode?: { node: string; error: string }): void {
    if (workdir === undefined) return
    // A run a gate's reloop limit stopped says so, and how to go on, on every write.
    const detail = given ?? (status === 'halted' ? limitMessage(spine, workdir) : undefined)
    try {
      const dir = join(workdir, '.refactory')
      mkdirSync(dir, { recursive: true })
      // The cursor travels with the status: a reader of a finished run wants the
      // score, and a reader of an abandoned one wants to know which nodes never ran.
      const state = spine.state()
      const next = spine.next()
      const views = next.status === 'ready' ? next.nodes : []
      const runningNodes = views.map(v => ({ id: v.id, kind: nodeKind(v._type) }))
      const nodeList = runningNodes.map(n => `${n.id} (${n.kind})`).join(', ')
      const wait = status === 'paused' ? 'paused by user'
        : status === 'halted' && detail !== undefined ? detail
        : status === 'waiting_user' || views.some(node => node._type === 'GateNode' && node.evaluator_type === 'user')
          ? `waiting for user gate: ${nodeList || 'user gate'}`
          : views.some(node =>
              (node._type === 'AgentNode' && !isDeterministic(node))
              || node._type === 'LLMNode'
            ) ? `waiting for worker delivery: ${nodeList}`
            : status === 'executing'
              ? (nodeList ? `running a graph check: ${nodeList}` : 'running a graph check')
              : undefined
      // Preserve completion_override when the new status AND invocation both match;
      // clear it when either changes (#347, C1).
      let completionOverride: Record<string, unknown> | undefined
      try {
        const existing = JSON.parse(readFileSync(join(dir, 'spine-status.json'), 'utf8'))
        const prior = existing?.completion_override
        if (prior !== null && typeof prior === 'object' && !Array.isArray(prior)
            && typeof prior.confirmed_status === 'string'
            && prior.confirmed_status === status) {
          // Also check invocation — a resumed or new invocation clears the override
          const overrideInvocation = (prior as Record<string, unknown>).invocation ?? null
          let currentInvocation: unknown = null
          try {
            currentInvocation = JSON.parse(readFileSync(join(dir, 'invocation.json'), 'utf8')).id ?? null
          } catch { /* no invocation file */ }
          if (overrideInvocation === currentInvocation) {
            completionOverride = prior as Record<string, unknown>
          }
        }
      } catch { /* absent file — no override to preserve */ }
      // The runtime and the owning process travel with the status: the walker writes
      // this file too (`run_state.write_status`), and a reader asks which one ran it.
      const record = {
        schema_version: 1, run_id: resolve(workdir), runtime: 'spine', owner_pid: process.pid,
        at: new Date().toISOString(),
        status, nodes: state.cursor, running_nodes: runningNodes, completed: state.completed,
        ...(detail === undefined ? {} : { error: detail }),
        ...(wait === undefined ? {} : { wait_reason: wait }),
        ...(status === 'paused' ? { paused_before: spine.pausedAt() } : {}),
        ...(failedNode !== undefined ? {
          last_failure: { node: failedNode.node, error: failedNode.error, at: new Date().toISOString() },
        } : {}),
        ...(completionOverride !== undefined ? { completion_override: completionOverride } : {}),
      }
      const path = join(dir, 'spine-status.json')
      writeFileSync(`${path}.tmp`, JSON.stringify(record, null, 2))
      renameSync(`${path}.tmp`, path)
      const checkpoint = join(dir, 'spine-checkpoint.json')
      const saved = spine.checkpoint()
      const graph = saved.graph as WorkflowGraph
      const paths = new Set(Object.values(graph.nodes).flatMap(node => [...node.reads ?? [], ...node.writes ?? []]))
      if (existsSync(join(dir, 'run-config.json'))) paths.add('.refactory/run-config.json')
      const file_fingerprints = Object.fromEntries([...paths].map(path => [path, fingerprint(resolve(workdir, path))]))
      writeFileSync(`${checkpoint}.tmp`, JSON.stringify({ ...saved, status, owner_pid: process.pid,
        file_fingerprints, materialised: materialised.get(spine) ?? null, run_id: resolve(workdir),
        graph_revision: computeGraphRevision(saved) }))
      renameSync(`${checkpoint}.tmp`, checkpoint)
      appendFileSync(join(dir, 'events.jsonl'), JSON.stringify({ ...record, kind: 'graph_state' }) + '\n')

    } catch (error) {
      // A stale ready checkpoint could replay a command whose outcome is unknown.
      // Persist the execution marker before dispatch, and stop on every write error.
      throw new Error('Cannot persist graph recovery checkpoint; continuation refused', { cause: error })
    }
  }

  /**
   * The gate card's line on how many reloops are left before a reloop stops the run
   * (dm-141), or undefined for a gate without a counted reloop.
   * @param spine - the run's spine.
   * @param gate - the user gate being asked.
   * @returns the line.
   */
  function reloopsLeft(spine: Spine, gate: string): string | undefined {
    const spent = spine.reloops(gate)
    if (spent === undefined) return undefined
    const left = Math.max(0, spent.cap - spent.used)
    return left === 0
      ? `Reloops left: 0 of ${spent.cap}. A reloop now stops the run instead of going back to ${spent.target}.`
      : `Reloops left: ${left} of ${spent.cap}.`
  }

  /**
   * What a person reads when a gate's reloop limit stopped the run: that the limit was
   * reached, and how to go on (dm-141). Undefined for any other halt.
   * @param spine - the run's spine.
   * @param workdir - the run directory, named in the commands.
   * @returns the message.
   */
  function limitMessage(spine: Spine, workdir: string | undefined): string | undefined {
    const halted = spine.state().halted as { gate?: string; reason?: string } | undefined
    if (halted?.reason !== 'max_iterations' || !halted.gate) return undefined
    const spent = spine.reloops(halted.gate)
    const limit = spent === undefined ? 'its reloop limit' : `its limit of ${spent.cap} reloops`
    const target = spent?.target ?? 'the node it reloops to'
    const run = workdir ?? 'RUN_DIR'
    return `The run stopped: ${halted.gate} reached ${limit}. To go on, rerun ${target}: in the GUI, `
      + 'choose it, then Preview affected results and Confirm correction and rerun; or run '
      + `\`refactory graph-control preview --run ${run} --node ${target}\` and then `
      + `\`refactory graph-control rerun --run ${run} --node ${target} --revision REVISION\` `
      + 'with the revision it returns. A rerun by a person resets the limit.'
  }

  /** The status {@link recordStatus} last wrote for the run, or undefined when there is none to read. */
  function recordedStatus(workdir: string | undefined): string | undefined {
    if (workdir === undefined) return undefined
    try {
      const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')) as { status?: unknown }
      return typeof recorded?.status === 'string' ? recorded.status : undefined
    } catch {
      return undefined
    }
  }

  /** The full status record {@link recordStatus} last wrote, or undefined when absent. */
  function readStatusRecord(workdir: string | undefined): Record<string, unknown> | undefined {
    if (workdir === undefined) return undefined
    try {
      const raw = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
      return raw !== null && typeof raw === 'object' && !Array.isArray(raw) ? raw as Record<string, unknown> : undefined
    } catch {
      return undefined
    }
  }

  /** Consume only controls addressed to this exact durable invocation. */
  function applyControls(spine: Spine, workdir: string | undefined, env: Record<string, string>, dataCommand: string, _agentCommand?: string): void {
    if (workdir === undefined) return
    const directory = join(workdir, '.refactory', 'graph-controls')
    if (!existsSync(directory)) return
    const invocationPath = join(workdir, '.refactory', 'invocation.json')
    const invocation = existsSync(invocationPath) ? JSON.parse(readFileSync(invocationPath, 'utf8')).id : null
    for (const name of readdirSync(directory).filter(name => name.endsWith('.json')).sort()) {
      const path = join(directory, name)
      const request = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
      const receipt = { ...request, status: 'applied', message: 'Applied at graph boundary; continuation requires the owning driver.' }
      try {
        if (request.run_id !== resolve(workdir) || request.invocation !== invocation) throw new Error('control belongs to another run invocation')
        const state = spine.state()
        if (request.action === 'pause') {
          if (typeof request.target === 'string' && request.target !== '') {
            spine.pauseBefore(String(request.target), true)
          } else {
            const cursor = state.cursor as string[]
            const targets = cursor.length ? cursor : spine.passOf() ? [spine.passOf()!.node] : []
            for (const node of targets) spine.pauseBefore(node, true)
          }
        }
        else if (request.action === 'resume') for (const node of state.pause_before as string[]) spine.pauseBefore(node, false)
        else if (request.action === 'stop') { spine.stopAtBoundary(); receipt.message = 'Stopped at graph boundary; no further nodes will dispatch.' }
        else if (request.action === 'rerun') {
          const target = String(request.target)
          const isCommandCorrection = request.source === 'amend-command'
          const deferredNodes = request.deferred_nodes as string[] | undefined
          // dm-227: a person's rerun reopens work behind their own pause (the
          // `whilePaused` below, which the CEO's wf_redo never passes), and a run that
          // was paused when it arrived then holds at the reopened node.
          const wasPaused = spine.isPaused()

          if (Array.isArray(deferredNodes) && deferredNodes.length > 0) {
            if (spine.hasConcurrentWork()) {
              // Concurrent work still running — skip this control, keep file.
              receipt.status = 'deferred'
              receipt.message = 'Concurrent work still running; control retained for next boundary.'
              writeFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), JSON.stringify(receipt))
              appendFileSync(join(workdir, '.refactory', 'events.jsonl'), JSON.stringify({ schema_version: 1,
                run_id: resolve(workdir), invocation, at: new Date().toISOString(), kind: 'control',
                state: receipt.status, summary: receipt.message }) + '\n')
              continue  // skip unlinkSync — file survives to next boundary
            }
            // Concurrent work has finished. Verify the plan still matches.
            const currentPlan = spine.planRedo(target, { whilePaused: true })
            const currentSet = new Set(currentPlan.nodes)
            const deferredSet = new Set(deferredNodes)
            if (currentSet.size !== deferredSet.size
                || ![...currentSet].every(n => deferredSet.has(n))) {
              throw new Error(
                'invalidation plan changed since preview; re-preview before rerunning'
              )
            }
            // Plan check passed — skip the revision check for deferred controls.
          } else {
            // Non-deferred: compare the recomputed graph revision from the
            // live spine's checkpoint and state. Never compare against a
            // stored field; drop the byte-hash fallback so a pre-upgrade
            // preview is refused with the existing "preview again" message.
            const liveSaved = spine.checkpoint()
            const liveRevision = computeGraphRevision(liveSaved)
            if (liveRevision !== request.revision) throw new Error('graph changed since preview; preview invalidation again')
          }
          const plan = spine.planRedo(target, { commandCorrection: isCommandCorrection, whilePaused: true })
          const nodes = (spine.checkpoint().graph as WorkflowGraph).nodes
          if (typeof request.text === 'string' && request.text.trim() !== '' && !['AgentNode', 'LLMNode'].includes(nodes[target]?._type ?? '')) {
            throw new Error('this node has no editable prompt; correct its source with explicit invalidation')
          }
          if (dataCommand !== '') {
            recordStatus(spine, workdir, 'executing')
            runtimeBridge(`${dataCommand} reset`, workdir, env, { ...plan, state, archiveArtifacts: true })
          }
          if (isCommandCorrection) {
            spine.redoCommandCorrection(target, { whilePaused: true })
          } else {
            spine.redo(target, { whilePaused: true })
          }
          // dm-227: a rerun applied while the run was paused holds before the redo's
          // start node until the person resumes. A live run is not held, even when a
          // targeted pause is still armed on a route it never took.
          if (wasPaused) spine.pauseBefore(plan.start, true)
          const gateReason = spine.gateReasonFor(target)
          const rerunText = typeof request.text === 'string' && request.text.trim() !== '' ? request.text.trim() : ''
          if (gateReason !== undefined || rerunText !== '') {
            const parts = [gateReason, rerunText].filter(Boolean)
            spine.setFeedback(target, parts.join('\n\n'))
          }
          if (plan.resetPass) materialised.delete(spine)
          receipt.message = `Reopened ${target}; invalidated ${plan.nodes.join(', ')}; prior evidence archived.`
        } else if (request.action === 'skip') {
          const target = String(request.target)
          const provenance = { source: 'person' as const, at: String(request.at || new Date().toISOString()) }
          spine.skip(target, provenance)
          receipt.message = `Skipped ${target} (person).`
        } else throw new Error('unsupported graph control; supported: pause, resume, stop, rerun, skip')
      } catch (error) {
        receipt.status = 'failed'
        receipt.message = error instanceof Error ? error.message : String(error)
        recordWarnings(workdir, [{ node: '__graph_control__', line: `graph control ${String(request.action)} failed: ${receipt.message}` }])
      }
      if (receipt.status === 'applied') recordStatus(spine, workdir,
        spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : spine.next().status)
      if (receipt.status !== 'deferred') {
        writeFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), JSON.stringify(receipt))
        appendFileSync(join(workdir, '.refactory', 'events.jsonl'), JSON.stringify({ schema_version: 1,
          run_id: resolve(workdir), invocation, at: new Date().toISOString(), kind: 'control',
          state: receipt.status, summary: receipt.message }) + '\n')
        unlinkSync(path)
      }
    }
  }

  /**
   * Journal the run warnings one dispatch boundary's nodes printed.
   *
   * The `wf_next` view hands them to the CEO, but a CEO can end its turn without
   * passing them on, and a headless run has nobody reading its turns. The journal
   * outlives the turn: the run's final report and the GUI's activity history read
   * it from there.
   * @param workdir - the run directory whose journal records them.
   * @param printed - each warning line, with the id of the node that printed it.
   */
  function recordWarnings(workdir: string | undefined, printed: Array<{ node: string; line: string }>): void {
    if (workdir === undefined || printed.length === 0) return
    const refDir = join(workdir, '.refactory')
    mkdirSync(refDir, { recursive: true })
    const invocationPath = join(refDir, 'invocation.json')
    const invocation = existsSync(invocationPath) ? JSON.parse(readFileSync(invocationPath, 'utf8')).id : null
    const at = new Date().toISOString()
    appendFileSync(join(refDir, 'events.jsonl'), printed.map(({ node, line }) => JSON.stringify({
      schema_version: 1, run_id: resolve(workdir), invocation, at, kind: 'run_warning',
      node_id: node, state: 'warning', summary: line,
    }) + '\n').join(''))
  }

  /**
   * Run the deterministic nodes the cursor is sitting on, so `wf_next` reports
   * only work a model must do. A node that fails leaves the cursor where it is
   * and is reported instead: the graph must not advance past a step that did
   * not happen.
   *
   * A data node is iterated here too. The cursor cannot enter a subgraph on its
   * own, so the loop selects the node's items, opens a pass, walks the subgraph the
   * same way it walks the rest of the graph, and closes the pass when its exit
   * completes — which is what makes one pass land per item on this runtime, the way
   * the walking executor gives it.
   *
   * With `interactiveOpen` the loop stops after **one dispatch boundary** and
   * returns, so each `wf_next` is one step the CEO can watch and steer between.
   * Every deterministic node the cursor holds at once is a fork peer the graph
   * declared concurrent, so one boundary dispatches the whole ready set together
   * (#306) rather than one node at a time. Static fork/join boundaries and
   * opening/closing a data pass are bookkeeping,
   * not executable nodes, so they do not consume that one-boundary budget. A pass
   * that begins always continues on
   * the calls that follow. A configured graph (`interactiveOpen: false`) keeps
   * draining every automatic node in one call.
   * @param spine - the run to advance.
   * @param workdir - the run directory node commands run in.
   * @param commandEnv - the run's own environment for node commands.
   * @param dataCommand - the run's own item loader command.
   * @param warnings - collects the run warnings every boundary this call ran printed,
   * each led by its node's id, including those of a boundary that then failed.
   */
  async function drain(
    spine: Spine,
    workdir: string | undefined,
    commandEnv: Record<string, string>,
    dataCommand: string,
    agentCommand: string,
    warnings: string[],
  ): Promise<{ next: SpineNext; failure?: CommandFailure; item?: unknown; ran?: RanStep | undefined }> {
    if (!config.executeCommands) return { next: spine.next() }
    // Structural boundaries have no worker or command to dispatch. Cross a sole
    // due boundary before exposing branch prompts, so their delivery snapshots
    // exist before a CEO can delegate them. Mixed cursors retain manual control:
    // completing a fork there would replace its concurrent peers.
    const structural = new Set<string>()
    const advanceControl = (state: SpineNext): boolean => {
      if (state.status !== 'ready' || spine.isPaused() || state.nodes.length !== 1) return false
      const node = state.nodes[0]!
      if (node._type !== 'ForkNode' && node._type !== 'JoinNode') {
        structural.clear()
        return false
      }
      if (structural.has(node.id)) throw new Error(`structural workflow cycle at ${node.id}`)
      structural.add(node.id)
      spine.complete(node.id)
      return true
    }
    const shell = ctx.get('shell')
    if (shell === undefined) {
      // Without an executor nothing deterministic can run. That is only a problem
      // when something is waiting to be run: a data node handed to the driving
      // agent would be a node the cursor never enters and never iterates.
      let state = spine.next()
      while (advanceControl(state)) state = spine.next()
      const waiting = state.status === 'ready'
        ? state.nodes.find(node => node._type === 'DataNode')
        : undefined
      if (waiting === undefined) return { next: state }
      return {
        next: state,
        failure: {
          ok: false,
          command: waiting.id,
          nodeId: waiting.id,
          exitCode: null,
          stdout: '',
          stderr: '',
          error: `${waiting.id} iterates data and this composition mounts no shell executor to load its items`,
        },
      }
    }
    const runner = {
      env: commandEnv,
      timeoutMs: config.commandTimeoutMs,
      dataCommand,
      sandboxPolicy: { mode: config.commandSandboxMode as 'danger-full-access', workspaceRoot: workdir ?? process.cwd() },
      ...(workdir === undefined ? {} : { workdir }),
    }
    // The node this call ran, when the composition stops after one. A configured
    // graph runs many per call and reports none of them individually.
    let ran: RanStep | undefined
    let item: unknown

    const recordBoundary = (): void => recordStatus(spine, workdir,
      spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : spine.next().status)

    /** Select a data node's items and open its first pass. */
    const openPass = async (data: NodeView): Promise<{ failure?: CommandFailure }> => {
      recordStatus(spine, workdir, 'executing')
      const listed = await runDataCommand(shell, data, 'list', runner)
      if (!listed.ok) return { failure: listed }
      const count = Number(listed.value.count ?? 0)
      // An empty selection is a fact, not a pass: the node completes and the run
      // continues, exactly as the walking executor continues past it.
      if (count === 0) {
        spine.complete(data.id)
        recordBoundary()
        return {}
      }
      spine.startPass(data.id, count)
      recordBoundary()
      return {}
    }

    /** Materialise the running pass's item, once per item across calls. */
    const beginItem = async (): Promise<{ failure?: CommandFailure }> => {
      const pass = spine.passOf()
      if (pass === undefined) return {}
      const key = `${pass.node}:${pass.index}`
      const current = materialised.get(spine)
      if (current?.key === key) {
        item = current.item
        return {}
      }
      const data = spine.view(pass.node)
      recordStatus(spine, workdir, 'executing')
      const started = await runDataCommand(shell, data, 'begin', runner, pass.index)
      if (!started.ok) return { failure: started }
      materialised.set(spine, { key, item: started.value })
      item = started.value
      recordBoundary()
      return {}
    }

    /** Close a finished pass: judge it, then move to the next item or finish. */
    const close = async (): Promise<{ failure?: CommandFailure }> => {
      const pass = spine.passOf()
      if (pass === undefined || spine.isHalted()) return {}
      if (!pass.finished) {
        const pending = spine.next()
        // A terminal refusal ends this item's branch too. The Task judges its
        // missing answer, just as on the walker; it must not strand the dataset.
        if (pending.status !== 'ready' || pending.nodes.length !== 0) return {}
      }
      // Explicit skips may exhaust a body before any executable node is exposed.
      // Its Task still owns setup-before-verify, exactly once for this item.
      const started = await beginItem()
      if (started.failure !== undefined) return started
      applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
      if (spine.isHalted() || spine.isPaused()) return {}
      if (JSON.stringify(spine.passOf()) !== JSON.stringify(pass)) return {}
      const data = spine.view(pass.node)
      recordStatus(spine, workdir, 'executing')
      const ended = await runDataCommand(shell, data, 'end', runner, pass.index)
      if (!ended.ok) return { failure: ended }
      materialised.delete(spine)
      item = undefined
      spine.endPass()
      structural.clear()
      recordBoundary()
      return {}
    }

    for (;;) {
      applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
      if (spine.isHalted() || spine.isPaused()) return { next: spine.next(), item, ran }
      const closed = await close()
      if (closed.failure !== undefined) return { next: spine.next(), failure: closed.failure, ran }
      let state = spine.next()
      if (state.status === 'done' || spine.isPaused()) return { next: state, item, ran }
      const started = await beginItem()
      if (started.failure !== undefined) return { next: state, failure: started.failure, ran }
      applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
      state = spine.next()
      if (spine.isHalted() || spine.isPaused()) return { next: state, item, ran }
      // Normalizing a standing skip can exhaust the next item immediately too.
      if (spine.passOf()?.finished) continue
      if (state.status === 'done') return { next: state, item, ran }
      if (advanceControl(state)) continue
      const data = state.nodes.find(node => node._type === 'DataNode')
      if (data !== undefined) {
        const openedPass = await openPass(data)
        if (openedPass.failure !== undefined) return { next: state, failure: openedPass.failure, ran }
        continue
      }
      const automatic = state.nodes.filter(isDeterministic)
      if (automatic.length === 0 || (interactiveOpen && ran !== undefined)) return { next: state, item, ran }
      // One dispatch boundary per call is the interactive contract; the configured
      // graph keeps walking every automatic node the cursor is sitting on.
      // Re-enter the boundary after each command, including concurrent cursors:
      // a queued stop/pause must take effect before any later node dispatch.
      //
      // Every node the cursor holds at once is a fork peer the graph declared
      // concurrent, so the whole ready set is dispatched together rather than one
      // at a time (#306): a serial fan pays the summed wall of independent
      // children. A queued stop/pause takes effect between dispatch boundaries,
      // not inside one. A node that fails is reported after its peers land: their
      // cursor positions complete, and the failure names the node that did not.
      // `maxConcurrentNodes` bounds how many of those peers hold a slot at once:
      // the shared resources behind the commands — one model gateway, one host —
      // are not free, so a composition caps the fan and queues the rest in cursor
      // order, a slot freeing as each command settles rather than in waves.
      recordStatus(spine, workdir, 'executing')
      const cap = config.maxConcurrentNodes ?? 0
      const ceiling = cap > 0 ? cap : Number.POSITIVE_INFINITY
      let inFlight = 0
      const waiters: Array<() => void> = []
      // dm-110: write feedback sidecar files for dispatched agent nodes.
      const feedbackMap = spine.next().feedback ?? {}
      const feedbackFiles: Record<string, string> = {}
      if (workdir !== undefined) {
        const promptDir = join(workdir, '.refactory', 'prompts')
        for (const node of automatic) {
          const fbText = feedbackMap[node.id]
          if (fbText !== undefined && fbText.trim() !== '') {
            mkdirSync(promptDir, { recursive: true })
            const fbFile = join(promptDir, `${node.id}.feedback.md`)
            // Atomic write: temporary file then rename, so a stale file is never read.
            const fbTmp = join(promptDir, `${node.id}.feedback.tmp`)
            writeFileSync(fbTmp, fbText)
            renameSync(fbTmp, fbFile)
            feedbackFiles[node.id] = fbFile
          }
        }
      }
      const outcomes = await Promise.all(automatic.map(async node => {
        // A finishing command hands its slot straight to the first waiter rather
        // than returning it to the pool, so a fresh starter cannot slip in
        // between the release and the waiter's wakeup and overshoot the ceiling.
        if (inFlight >= ceiling) await new Promise<void>(resolve => waiters.push(resolve))
        else inFlight += 1
        try {
          return await runDeterministic(shell, node, runner, feedbackFiles[node.id])
        } finally {
          const next = waiters.shift()
          if (next === undefined) inFlight -= 1
          else next()
        }
      }))
      let failure: CommandFailure | undefined
      for (let index = 0; index < automatic.length; index += 1) {
        const node = automatic[index]!
        const outcome = outcomes[index]!
        if (!outcome.ok) {
          if (failure === undefined) failure = outcome
          continue
        }
        // A gate's reason is the reloop target's feedback, exactly as a manual
        // verdict's is: without it the node the run goes back to, and the CEO,
        // see an empty string where the cause should be.
        if (outcome.verdict === undefined) spine.complete(node.id)
        else spine.verdict(node.id, outcome.verdict, outcome.reason ?? '')
      }
      recordBoundary()
      // Every peer's warnings, not only a named node's: a warning is about the
      // run, and the peers are otherwise reported by id alone. The journal keeps a
      // failed node's too -- a warning matters most on a run that went wrong -- while
      // this call's view leaves them to the failure, whose stderr already carries them.
      const printed = automatic.flatMap((node, index) => {
        const outcome = outcomes[index]!
        return runWarningsOf(outcome.stderr).map(line => ({ node: node.id, line, ok: outcome.ok }))
      })
      recordWarnings(workdir, printed)
      const boundaryWarnings = printed.filter(({ ok }) => ok).map(({ node, line }) => `${node}: ${line}`)
      warnings.push(...boundaryWarnings)
      if (failure !== undefined) return { next: spine.next(), failure, ran }
      if (interactiveOpen) {
        // Preserve the one-boundary budget while allowing structural boundaries
        // and data-pass grading to settle before exposing the next ready work.
        const last = automatic.length - 1
        ran = recordedStep(automatic[last]!, workdir, automatic, outcomes[last])
        if (boundaryWarnings.length > 0) ran.warnings = boundaryWarnings
      }
    }
  }

  /**
   * The `wf_next` payload: the due nodes, or the node that failed to run.
   * @param spine - the run to advance.
   * @param workdir - the run directory its status is written to.
   * @param commandEnv - the run's own environment for node commands.
   * @param dataCommand - the run's own item loader command.
   */
  async function report(
    spine: Spine,
    workdir: string | undefined,
    commandEnv: Record<string, string>,
    dataCommand: string,
    agentCommand: string,
  ): Promise<JsonValue> {
    applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
    /** Drain pending skip receipts through the runtime bridge. */
    function drainSkipReceipts(s: Spine, wd: string | undefined, env: Record<string, string>, cmd: string): void {
      if (wd === undefined || cmd === '') return
      const state = s.state()
      const pending = state.pending_skip_receipts as Array<{ node: string; provenance: { source: string; at: string } }> | undefined
      if (!pending || pending.length === 0) return
      for (const entry of [...pending]) {
        try {
          runtimeBridge(cmd, wd, env, {
            action: 'record-skip', node: entry.node, provenance: entry.provenance,
          })
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error)
          recordStatus(s, wd, 'failed', `record-skip for ${entry.node} failed: ${detail}`,
            { node: entry.node, error: `record-skip failed: ${detail}` })
          throw new Error(`record-skip for ${entry.node} failed: ${detail}`)
        }
      }
      // Clear after successful drain: mutate pending_skip_receipts in place via the
      // spine's internal queue (accessed through skip receipts being drained).
      // We need to call state() again to clear — but since pendingSkipReceipts is
      // exposed on the spine's state, we clear it by checkpoint/restore or by the
      // spine clearing its internal array. For now, clear via a small accessor.
      ;(s as any).pendingSkipReceipts = []
    }
    drainSkipReceipts(spine, workdir, commandEnv, agentCommand)
    let drained: Awaited<ReturnType<typeof drain>>
    const warnings: string[] = []
    try { drained = await drain(spine, workdir, commandEnv, dataCommand, agentCommand, warnings) }
    catch (error) {
      recordStatus(spine, workdir, 'failed', error instanceof Error ? error.message : String(error))
      throw error
    }
    const { next, failure, item, ran } = drained
    // Warnings no `ran` step carries travel on the view itself: a configured graph
    // drains many boundaries per call and names none of them, and a failed boundary
    // is reported as the failure, not as a step.
    const unreported = ran === undefined && warnings.length > 0 ? { warnings } : {}
    if (failure !== undefined) {
      // The node's own stderr is why it failed. The exit code alone sends a reader
      // back to re-run the node to find out, so the reason is recorded with it.
      // dm-95: write status ready (not failed) with last_failure — a node failure
      // the graph may retry is not a finished, failed run. The caller (CEO) still
      // sees status: 'failed' so it knows what happened. Exit code 3 with the
      // refactory failure marker is a build-loop limit stop (dm-202): that records
      // 'failed' (terminal) while other nonzero codes remain 'ready' (retryable).
      const reason = failure.stderr.trim()
      const failDetail = reason === '' ? failure.error : `${failure.error}: ${failureDetailOf(reason)}`
      const isLimitStop = failure.exitCode === 3 && reason.includes('--- refactory failure ---')
      recordStatus(spine, workdir,
        isLimitStop ? 'failed'
          : spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : 'ready',
        undefined,
        { node: failure.nodeId, error: failDetail })
      return json({
        status: 'failed',
        node: failure.nodeId,
        command: failure.command,
        exit_code: failure.exitCode,
        error: failDetail,
        stdout: failure.stdout,
        stderr: failure.stderr,
        ...unreported,
      })
    }
    drainSkipReceipts(spine, workdir, commandEnv, agentCommand)
    if (next.status === 'ready' && !spine.isPaused() && agentCommand !== '' && workdir !== undefined) {
      const runKey = workdir
      for (const node of next.nodes) {
        if (node._type !== 'AgentNode' && node._type !== 'LLMNode') continue
        // Dispatched AgentNodes are run by commandOf()/drain(); skip them here
        // to avoid recording a double delivery.
        if (isDeterministic(node)) continue
        const receipt = runtimeBridge(agentCommand, workdir, commandEnv, {
          action: 'begin', node: node.id, prompt: node.prompt_template ?? '',
          reads: node.reads ?? [], writes: node.writes ?? [],
        })
        if (typeof receipt.prompt !== 'string') throw new Error('agent delivery bridge returned no prompt')
        node.prompt_template = receipt.prompt
        const mapKey = `${runKey}:${node.id}`
        // Clear old CEO prompts only on fresh dispatch (not reuse), so parallel
        // reviewers do not lose each other's recorded prompts (#347, C2).
        const isNewDispatch = !pendingPrompts.has(mapKey)
        pendingPrompts.set(mapKey, receipt.prompt)
        if (isNewDispatch) {
          ceoPrompts.delete(mapKey)
        }
        // Record unresolved template keys as a run warning and cache for receipt (#347, C2).
        // Warn once per delivery, not per wf_next call.
        const unresolvedKeys = receipt.unresolved_keys
        if (Array.isArray(unresolvedKeys) && unresolvedKeys.length > 0) {
          if (!unresolvedKeyWarned.has(mapKey)) {
            unresolvedKeyWarned.add(mapKey)
            recordWarnings(workdir, unresolvedKeys.map((k: unknown) => ({
              node: node.id,
              line: `unresolved template key: {{${String(k)}}}`,
            })))
          }
          pendingUnresolvedKeys.set(mapKey, unresolvedKeys as unknown[])
        }
      }
    }
    // A user gate's question is shown as the CEO sees it here and as the card puts it.
    if (next.status === 'ready') {
      for (const node of next.nodes) {
        if (node._type === 'GateNode' && node.gate_prompt !== undefined) {
          node.gate_prompt = withFacts(node.gate_prompt, workdir)
          // Append CEO-sourced ledger notes at the gate view, labelled as the CEO's.
          if (config.steerCommand !== '' && workdir !== undefined) {
            try {
              const steerResult = spawnSync(config.steerCommand, [...config.steerArgs, workdir], {
                encoding: 'utf8', timeout: config.commandTimeoutMs,
              })
              if (steerResult.status === 0) {
                const ledger = JSON.parse(steerResult.stdout)
                const entries = Array.isArray(ledger?.entries) ? ledger.entries : []
                const ceoNotes = entries
                  .filter((e: any) => e?.source === 'ceo' && typeof e?.text === 'string')
                  .map((e: any) => e.text)
                if (ceoNotes.length > 0) {
                  const ceoNotesText = ceoNotes.join('\n\n')
                  ;(node as any).ceo_notes = ceoNotesText
                  node.gate_prompt += '\n\n--- CEO observations (not requirements) ---\n' + ceoNotesText
                }
              }
            } catch { /* guard: do not block the gate */ }
          }
        }
      }
    }
    // A run that stopped short of its end says which nodes it never reached.
    const halted = next.status === 'done' ? next.halted : undefined
    const unreached = halted?.reason === 'incomplete' ? `never reached: ${(halted.unreached ?? []).join(', ')}` : undefined
    recordStatus(spine, workdir, spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : next.status === 'done' ? 'done' : 'ready', unreached)
    // The item being iterated travels with the node views, so the subagent for a
    // pass is handed the item rather than being sent to find it on disk. Under
    // `interactiveOpen` the node this call ran travels the same way, so the CEO
    // can say which one it was and what it left behind.
    const current = spine.isPaused() ? { ...next, status: 'paused' } : next
    const view = item === undefined ? current : { ...current, item }
    return json(ran === undefined ? { ...view, ...unreported } : { ...view, ran })
  }

  if (configuredSpine !== undefined) {
    const spine = configuredSpine
    const nodes = Object.values(config.graph?.nodes ?? {})
    // drain deliberately leaves mixed structural cursors to a driver. Keep
    // multiple/nested forks on the CEO path rather than partially executing them.
    const flatControls = nodes.filter(node => node._type === 'ForkNode').length <= 1
      && nodes.filter(node => node._type === 'JoinNode').length <= 1
    const automatic = config.executeCommands && flatControls && nodes.length > 0 && nodes.every(node =>
      node.expand === undefined && (Boolean(commandOf(node)?.trim())
        || node._type === 'ForkNode' || node._type === 'JoinNode'
        || (node._type === 'DataNode' && config.dataCommand !== '')),
    )
    ctx.provide(CONFIGURED_RUN_SERVICE, {
      automatic,
      advance: () => report(spine, config.workdir, config.commandEnv, config.dataCommand, config.agentCommand ?? ''),
    } satisfies ConfiguredRun)
  }

  const pendingQuestions = new WeakSet<Spine>()

  /**
   * The verdict a person gives at a `user` gate, asked by the runtime.
   *
   * Asked here rather than left to the model's own `ask_user_question`, because a
   * model that can route a gate without asking can invent the answer, and an invented
   * verdict is indistinguishable from one a person gave. The gate's own outcome labels
   * are the options, so the answer is always one the graph accepts.
   * @param gate - the gate node id.
   * @param spine - the run whose gate is being routed.
   * @param view - the gate's view, whose `verdicts` are the offered options.
   * @param exec - the tool execution, for the calling agent and its abort signal.
   * @param workdir - the run directory a refusal is recorded in.
   * @returns the verdict the person chose, or no verdict when a graph control interrupted the question.
   */
  async function askPerson(
    gate: string,
    spine: Spine,
    view: NodeView,
    exec: ToolExec,
    workdir: string | undefined,
  ): Promise<UserGateResult> {
    if (pendingQuestions.has(spine)) throw new Error('a user-gate question is already pending for this run')
    pendingQuestions.add(spine)
    try {
      return await askCurrentReview(gate, spine, view, exec, workdir)
    } finally {
      pendingQuestions.delete(spine)
    }
  }

  async function askCurrentReview(
    gate: string, spine: Spine, view: NodeView, exec: ToolExec, workdir: string | undefined,
  ): Promise<UserGateResult> {
    const { gateCommand, commandEnv, dataCommand, agentCommand } = requireRun(exec)
    applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
    if (spine.isPaused() || spine.isHalted()) return undefined
    const request = { gate, reads: view.reads ?? [], labels: spine.acceptedVerdicts(gate) }
    // The engine halts the run at any gate, with or without a `halt` edge, so the
    // person is always offered it; otherwise "stop" is unanswerable at a user gate.
    const labels = request.labels.length === 0 || request.labels.includes(HALT)
      ? request.labels : [...request.labels, HALT]
    const prior = gateCommand !== '' && workdir !== undefined
      ? runtimeBridge(gateCommand, workdir, commandEnv, { ...request, action: 'begin' })
      : undefined
    if (typeof prior?.verdict === 'string' && prior.verdict !== '') return { verdict: prior.verdict, feedback: '' }
    const stateBefore = JSON.stringify(spine.state())
    const userQuestions = userQuestionsService()
    if (userQuestions === undefined) {
      const message = `gate ${gate} is answered by a person and this composition mounts no user-questions `
        + `channel, so it cannot be asked here; ${answerElsewhere(gate, labels, workdir)}`
      recordStatus(spine, workdir, 'refused', message)
      throw new Error(message)
    }
    let answer
    const questionLifetime = new AbortController()
    const questionSignal = exec.signal === undefined ? questionLifetime.signal : AbortSignal.any([exec.signal, questionLifetime.signal])
    try {
      recordStatus(spine, workdir, 'waiting_user')
      // The question card shows `question` as a one-line title and renders `detail` as
      // text, so the question is the gate's ask -- the prompt's last paragraph -- with the
      // reloops left, and what comes before it (design mode's card) and the CEO's notes
      // are the detail (dm-163, dm-141).
      const { ask, detail } = splitGatePrompt(withFacts(view.gate_prompt ?? '', workdir) || `Gate ${gate}`)
      const left = reloopsLeft(spine, gate)
      const questionText = left === undefined ? ask : `${ask} ${left}`
      const details = detail === undefined ? [] : [detail]
      if (config.steerCommand !== '' && workdir !== undefined) {
        try {
          const steerResult = spawnSync(config.steerCommand, [...config.steerArgs, workdir], {
            encoding: 'utf8', timeout: config.commandTimeoutMs,
          })
          if (steerResult.status === 0) {
            const ledger = JSON.parse(steerResult.stdout)
            const entries = Array.isArray(ledger?.entries) ? ledger.entries : []
            const ceoNotes = entries
              .filter((e: any) => e?.source === 'ceo' && typeof e?.text === 'string')
              .map((e: any) => e.text)
            if (ceoNotes.length > 0) {
              details.push('--- CEO observations (not requirements) ---\n' + ceoNotes.join('\n\n'))
            }
          }
        } catch { /* guard: do not block the gate */ }
      }
      const nativeAnswer = userQuestions.ask({
        questions: [{
          id: gate,
          header: 'Workflow gate',
          question: questionText,
          ...(details.length === 0 ? {} : { detail: details.join('\n\n') }),
          ...(labels.length === 0 ? {} : { options: labels.map(label => ({ label })) }),
        }],
        ...(exec.agent === undefined ? {} : { agent: exec.agent }),
        signal: questionSignal,
      })
      const panelAnswer = async (): Promise<Awaited<typeof nativeAnswer> | undefined> => {
        if (workdir === undefined) return new Promise(() => {})
        while (!questionSignal.aborted) {
          await wait(500, undefined, { signal: questionSignal })
          applyControls(spine, workdir, commandEnv, dataCommand, agentCommand)
          if (spine.isPaused() || spine.isHalted() || JSON.stringify(spine.state()) !== stateBefore) return undefined
          if (gateCommand === '') continue
          const polled = runtimeBridge(gateCommand, workdir, commandEnv, { ...request, action: 'poll', review: prior?.review })
          if (typeof polled.verdict === 'string' && polled.verdict !== '') {
            return { answers: [{ id: gate, selected: [polled.verdict], custom: '' }] } as Awaited<typeof nativeAnswer>
          }
        }
        throw new Error('question cancelled')
      }
      answer = await Promise.race([nativeAnswer, panelAnswer()])
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : String(error)
      const message = `gate ${gate} is answered by a person and could not be asked: ${detail}; `
        + answerElsewhere(gate, labels, workdir)
      recordStatus(spine, workdir, 'refused', message)
      throw new Error(message)
    } finally { questionLifetime.abort() }
    if (answer === undefined) return undefined
    if (JSON.stringify(spine.state()) !== stateBefore) {
      throw new Error(`gate ${gate} changed while awaiting the answer; ask again`)
    }
    const chosen = answer.answers.find(entry => entry.id === gate)
    if (chosen === undefined || (chosen.selected[0] ?? chosen.custom ?? '').trim() === '') {
      throw new Error(`gate ${gate} was answered without a verdict`)
    }
    const [picked] = chosen.selected
    let verdict: string
    let feedback = ''
    if (picked !== undefined) {
      if (labels.length > 0 && !labels.includes(picked.trim())) throw new Error(`gate ${gate} does not accept ${picked}`)
      verdict = picked.trim()
      // Thread the answerer's note through as feedback, so the gate bridge
      // can record it in the intent ledger.
      if (typeof chosen.custom === 'string' && chosen.custom.trim() !== '') {
        feedback = chosen.custom.trim()
      }
    } else {
      const typed = (chosen.custom ?? '').trim()
      const read = readGateAnswer(typed, labels)
      if (read === undefined) {
        // Anything but an outcome is conversation: the CEO replies, and the gate stays due.
        recordStatus(spine, workdir, 'ready')
        return { status: 'replied', gate, text: typed }
      }
      verdict = read.verdict
      feedback = read.feedback
    }

    if (gateCommand !== '' && workdir !== undefined) {
      runtimeBridge(gateCommand, workdir, commandEnv, {
        ...request, action: 'answer', review: prior?.review, verdict,
        ...(feedback ? { feedback } : {}),
      })
    }
    return { verdict, feedback }
  }

  // The driving contract. The spine enforces "what is next" structurally (it
  // hands over the next node only after the current one is reported complete),
  // and the deny-guard below removes the CEO's coding tools. This section tells
  // the CEO how to operate within those two constraints: advance with wf_next,
  // delegate each node's work to a subagent, report with wf_done, route gates.
  // A node's `reads`/`writes` are relative to the run directory, and a subagent's
  // working directory is *not* that directory — it is wherever the harness was
  // launched. Naming the run directory here, and requiring absolute paths in every
  // brief, is what keeps a node's work inside its run.
  const runDirectory = config.workdir === undefined
    ? 'the run directory the composition gave you'
    : `the run directory ${config.workdir}`
  // Only the run-time-open composition can be asked to start a mode, so only it is
  // told how. A composition with a configured graph has no `wf_open` to call.
  const openInstruction = interactiveOpen
    ? ` When the person asks you to run a mode and see each step, open it with ${toolPrefix}_open `
      + '(pass the mode as <package>/<name>), then drive it with '
      + `${toolPrefix}_next, reporting each node as it completes: each call runs one dispatch `
      + 'boundary — every deterministic node the cursor holds at once is a fork peer the graph '
      + 'declared concurrent, dispatched together — and names one under `ran` (with `peers` for '
      + 'the whole boundary) with the outputs and facts it left in the run '
      + 'directory, so say what it produced before acting on what is due next. When `ran` is a gate '
      + 'that sent the run back, its `verdict` and `reason` say why: tell the person, in a sentence, '
      + 'before driving the rerun. When `ran` carries `warnings` — for example that the target '
      + 'repository moved under the run — tell the person what each one says before going on. '
      + `Open any mode this way, `
      + 'including an authoring one (create/edit/design): it is the route where a person is present, and '
      + 'a gate marked `(user)` is put to them. For create/edit/design, include the full user brief as '
      + '`brief=<full text>` in the initial knobs array, alongside name/task/check knobs the mode declares. '
      + 'For design, `target=<repository root>` changes an existing repository, in a worktree of its own on the '
      + 'run\'s branch; without it the run starts a new project. Once its reviews pass, a design run opens a pull '
      + 'request, which its approval gate names first in `delivery.md` (`pr=0` keeps the work local); to follow up '
      + 'an open pull request, open design with `refine=<the pull request\'s URL>` and the follow-up as the brief. '
      + 'Intent records the same purpose in the authoritative ledger delivered to every agent node; '
      + 'an empty brief knob alone is not a reason to reopen a live run or edit a deterministic FnNode prompt. '
      + 'Use refactory action `runtime` when available to inspect inherited model/home context before asking setup questions. '
      + `A FnNode, Study, \`fn\` gate, or AgentNode with a dispatch command listed in \`nodes\` is deterministic`
      + ' — do NOT run its command and do NOT call '
      + `${toolPrefix}_done for it; call ${toolPrefix}_next again and the runtime runs it. `
      + `${toolPrefix}_done is only for a node whose work you or a subagent did.`
    : ''
  // Sacred Rules 8/9 and check-skip guardrail (#347, #363).
  const sacredRules = '\n\n'
    + 'You are an orchestrator. Delegate all technical work — writing code, fixing bugs, running checks, '
    + "performing research — to the graph's agent nodes. Review their output. When an agent fails, retry "
    + `with better instructions (${toolPrefix}_edit set-prompt/append-prompt (prefer append-prompt for nodes whose prompt contains template placeholders), then ${toolPrefix}_redo) or stop. Never take over `
    + "the agent's work yourself."
    + '\n\n'
    + "Every change that produces a pull request must pass through the graph's review nodes. "
    + '"The change is small" is not a reason to skip them.'
    + '\n\n'
    + 'Never claim, restate, or re-apply a person\'s verdict or approval in a ledger note '
    + 'or any other message. A user gate is answered only by the person through the runtime; '
    + 'a ledger note about a gate verdict is your observation, not an answer. '
    + `Never use ${toolPrefix}_edit skip on a check node (a node that produces runtime evidence). `
    + 'When a check cannot run, report what stopped it and what would unblock it. '
    + 'When the person wants to skip the check, give them the exact command: '
    + '`refactory graph-control skip --run <dir> --node <check_node>`. '
    + 'When the owner has stopped, the person can resume with '
    + '`refactory graph-control resume --run <dir> [--acknowledge-failure]`. '
    + 'If the person skips, the run report and PR say the check did not run.'
  // A configured graph runs many nodes per call and names none of them under `ran`,
  // so the warnings they printed come on the result itself and the CEO is told so.
  const warningInstruction = interactiveOpen
    ? ''
    : ` When a ${toolPrefix}_next result carries \`warnings\` — lines a step that ran printed for the `
      + 'person, for example that the target repository has uncommitted changes the run will not '
      + 'include — tell the person what each one says before going on, and repeat them in your final report.'
  ctx.systemPrompt.section({
    name: `tool:${toolPrefix}-spine`,
    order: ctx.systemPrompt.getSectionOrder('TOOL_WORKFLOW'),
    text: `You are the CEO of a workflow run, and you are DRIVING A WORKFLOW GRAPH. Your coding tools (bash, write, edit, web, skill) are disabled — you cannot implement, edit files, or browse yourself. Advance only through the wf tools.${openInstruction}${warningInstruction} If the person requests a pause before a named node or deterministic dataset, arm it FIRST with ${toolPrefix}_edit action \`pause\` for that node, before any ${toolPrefix}_next that could reach it. Use ${toolPrefix}_state to inspect ids without executing work. When status is \`paused\`, end your turn and wait for the person. Only after an explicit request to continue, use ${toolPrefix}_edit action \`unpause\` on that boundary and resume. Repeated ${toolPrefix}_next calls do not release a pause; never skip or redo around it. ${toolPrefix}_state inspects the cursor without execution; ${toolPrefix}_next executes due deterministic work. After reading steering and arming any requested pause, call it to advance, and call it again after every ${toolPrefix}_done / ${toolPrefix}_verdict / ${toolPrefix}_edit / ${toolPrefix}_expand — those tools only confirm the mutation, they do not tell you what is next. When ${toolPrefix}_steer is available, read it before each ${toolPrefix}_next: it is what the person said about the run since you last looked.${effectivePersonAttached ? ` When relaying the person's own words through ${toolPrefix}_steer, set source to 'user-relayed' so they become auditable requirements; your own observations default to 'ceo' and are context, not requirements.` : ''} A correction from the person is applied to that node's prompt with ${toolPrefix}_edit (\`set-prompt\`/\`append-prompt\`) and the node is re-run with ${toolPrefix}_redo. Prefer \`append-prompt\` for nodes whose prompt contains template placeholders (double-brace variables); \`set-prompt\` is refused when it would drop one. With automatic execution enabled, ${toolPrefix}_next crosses sole static ForkNode/JoinNode boundaries for you and establishes delivery contexts before returning the branch AgentNode/LLMNode prompts. Dispatch only those returned prompts in parallel; target ids in a state view are not dispatch instructions. If a ForkNode or JoinNode is explicitly returned (manual execution or a mixed cursor), acknowledge only that due control node with ${toolPrefix}_done, then call ${toolPrefix}_next before dispatching branch work. For an AgentNode or LLMNode (unless the runtime runs it — it has a dispatch command), spawn a subagent (the subagent tool) whose task is the node's prompt_template, and explicitly tell the subagent to read the node's reads files and write its result to the node's writes file (the filename in the node's writes field, e.g. candidate.py) — wait for it, then report the node with ${toolPrefix}_done. Every path in a node's reads and writes is relative to ${runDirectory}, and the subagent's own working directory is NOT that directory: give the subagent the absolute path (${runDirectory}/<the path>) for each file it must read or write, and tell it to write nowhere else. For a FnNode with a command, call ${toolPrefix}_next to execute it; only a command-less FnNode is yours to complete. Automatic gates also run through ${toolPrefix}_next; never supply their evaluator result. For a manually driven gate, use ${toolPrefix}_verdict with one of its listed outcomes. For a user gate the runtime asks the person itself and routes the answer they give: call ${toolPrefix}_verdict with the gate id and no verdict, and never answer for them — a verdict passed for their gate is refused. Before routing a user gate, read and summarise its \`reads\` files (listed in the node view returned by ${toolPrefix}_next) for the person — key decisions, scope, notable risks or non-goals — and link each file, so they see what they are deciding on before the question is put to them. Fan out a director's plan with ${toolPrefix}_expand. When a person says a step is wrong, read the run's ledger if one is configured, then use ${toolPrefix}_edit with \`set-prompt\` or \`append-prompt\` to correct that node's prompt and ${toolPrefix}_redo to run the node again with the correction. A DataNode is iterated for you: the runtime selects its items and hands you the subgraph's nodes one pass at a time, with the item being processed in the payload's item field and on disk as the node's current item file. Do not try to iterate it yourself, and do not report the data node — or the pass's last node — before its work is done. Never skip, reorder, or bypass a node — the runner refuses to advance until you report the current node complete.`
    + sacredRules,
  })

  // Open a mode's graph after boot, for a caller that will drive it node by node.
  // Registered only when `interactiveOpen` is set: a composition with a configured
  // graph has nothing to open, and a tool that cannot run must not be advertised.
  if (interactiveOpen) {
    ctx.tools.register(defineTool({
      name: `${toolPrefix}_open`,
      description: `Open a mode's graph and claim it for this session, so it can be driven node by node. Pass any mode the catalogue lists as \`<package>/<name>\` — a scored mode, or an authoring one ("create", "edit", "design") — and the open command lowers it and this reads the graph it wrote. This is the route where a person is present: the runtime puts a \`user\` gate to them with the person-questions service, and routes the answer they give, so a gate marked \`(user)\` is theirs and no verdict may be passed for it. Prefer this over a headless authoring call whenever someone can answer. Returns the post-drain view, so a deterministic start node has already run and is reported under \`ran\` rather than left due. One driven run at a time is open in a session: a second open is refused until that run finishes. All the other \`${toolPrefix}_*\` tools work on the opened run, and refuse when this session has not opened one.`,
      parameters: {
        mode: { type: 'string', required: true, description: 'The mode to run, as `<package>/<name>`.' },
        knobs: { type: 'array', items: { type: 'string' }, description: 'Knob overrides for this run, each as `NAME=VALUE` (repeatable). For create/edit/design, include `brief=<the full user brief>` on the first open, plus the other declared knobs such as name, task_ref or check; for design, `target=<repository root>` to change an existing repository (without it, the run starts a new project), `refine=<pull request URL>` to follow up an open pull request, and `pr=0` to open none.' },
        intent: { type: 'string', description: 'The user’s purpose for this run, recorded in its intent ledger. Without one the ledger records the requested mode identifier.' },
        workdir: { type: 'string', description: 'The run directory to lower into. When absent the spine derives one from this session and the mode — `<cwd>/.factory/web/<session>/<package>-<mode>` — creates it, and passes it as `--workdir`; the console\'s `open` verb requires a workdir, so one is always sent.' },
      },
      output: {
        schema: { type: 'json' },
        render: (_args, value) => [text(value)],
        presentationMeta: (_args, value) => presentationMeta('open', value),
      },
      presentCall: args => presentCall('open', args as Record<string, unknown>),
      presentResult: (args, result) => presentResult('open', args as Record<string, unknown>, result),
      execute(args, exec) {
        const session = exec.agent?.session
        if (session === undefined) {
          throw new Error(`${toolPrefix}_open requires an owning agent session`)
        }
        // Another chat's active or paused run does not block this session. A
        // completed run remains inspectable until its own session opens another.
        const owner = String(session.id)
        const opened = openedBySession.get(owner)
        if (opened !== undefined && opened.spine.next().status !== 'done') {
          throw new Error('a run is already open in this session')
        }
        if (config.openCommand === '') {
          throw new Error(
            `${toolPrefix}_open needs an open command; set \`openCommand\` (for example `
            + '`python -m refactory_dsh.console open`)',
          )
        }
        const argv = buildWfOpenArgv(config.openArgs, session.header?.cwd, args.mode)
        // The console refuses without a workdir, so an absent one is derived from
        // the session and the mode rather than dropped. The payload's `run` is
        // still the authority on where the run actually landed.
        const requested = args.workdir !== undefined && args.workdir !== ''
          ? args.workdir
          : derivedWorkdir(owner, args.mode)
        const requestedPath = canonicalWorkdir(requested)
        // Refuse before lowering, which can archive an existing invocation.
        // Terminal claims also remain readable; do not let another session's
        // next/state calls read or write a newly reused directory through them.
        for (const run of openedBySession.values()) {
          if (run.session === owner) continue
          const claimedPath = canonicalWorkdir(run.workdir)
          if (requestedPath === claimedPath
            || requestedPath.startsWith(claimedPath.endsWith(sep) ? claimedPath : claimedPath + sep)
            || claimedPath.startsWith(requestedPath.endsWith(sep) ? requestedPath : requestedPath + sep)) {
            throw new Error('the run directory overlaps a run owned by another session; choose a different workdir')
          }
        }
        argv.push('--workdir', requestedPath)
        // The run's workers inherit this session's model selection, so the person's
        // choice follows the run instead of the nested runtimes' own defaults. The
        // same selection the node-command `runEnv` reads, from the same sources:
        // the request header the next request will compare against, then the
        // agent's own options. An explicit project `refactory.workers.json` or
        // launch override still outranks this in the console's own precedence.
        const selection = session.requestHeader()?.config ?? exec.agent?.options
        if (typeof selection?.provider === 'string' && selection.provider !== '') argv.push('--provider', selection.provider)
        if (typeof selection?.model === 'string' && selection.model !== '') argv.push('--model', selection.model)
        for (const knob of args.knobs ?? []) argv.push('--knob', knob)
        if (args.intent !== undefined) argv.push(`--intent=${args.intent}`)
        const result = spawnSync(config.openCommand, argv, { encoding: 'utf8' })
        if (result.error !== undefined) {
          throw new Error(`${toolPrefix}_open could not run the open command: ${result.error.message}`)
        }
        if (result.status !== 0) {
          const reason = lastLine(result.stderr) || lastLine(result.stdout)
          throw new Error(
            `${toolPrefix}_open could not lower ${args.mode}: `
            + (reason === '' ? `the open command exited with ${String(result.status)}` : reason),
          )
        }
        let payload: OpenPayload
        try {
          payload = JSON.parse(result.stdout) as OpenPayload
        } catch {
          const said = lastLine(result.stdout)
          throw new Error(
            `${toolPrefix}_open: the open command printed no JSON payload (${said === '' ? 'no output' : said})`,
          )
        }
        // The command's payload is the authority on where the run landed; the
        // tool's own workdir is only the request. A command that answers with
        // neither leaves the run with nowhere to write its status, so it is
        // refused rather than opened half-addressed.
        const workdir = typeof payload.run === 'string' && payload.run !== ''
          ? payload.run
          : requested
        if (workdir === '') {
          throw new Error(`${toolPrefix}_open: the open command did not report a run directory`)
        }
        if (typeof payload.graph !== 'string' || payload.graph === '') {
          throw new Error(`${toolPrefix}_open: the open command did not report where the graph was lowered`)
        }
        let graph: WorkflowGraph
        try {
          graph = JSON.parse(readFileSync(payload.graph, 'utf8')) as WorkflowGraph
        } catch (error: unknown) {
          const detail = error instanceof Error ? error.message : String(error)
          throw new Error(`${toolPrefix}_open could not read the lowered graph at ${payload.graph}: ${detail}`)
        }
        const issues = validateGraph(graph)
        if (issues.length > 0) {
          const rendered = issues
            .map(issue => (issue.node === undefined ? issue.message : `${issue.node}: ${issue.message}`))
            .join('; ')
          throw new Error(`invalid workflow graph: ${rendered}`)
        }
        const spine = new Spine(graph, { maxIterations })
        restoreRequested(spine, workdir)
        const modelConfig = session.requestHeader()?.config ?? exec.agent?.options
        const runEnv = {
          ...(payloadEnv(payload.commandEnv) ?? config.commandEnv),
          // The parent owns these values: a lowered package cannot choose another
          // credential home. No key is copied into deterministic node commands.
          REFACTORY_MODEL_DSH_HOME: resolve((process.env.DSH_HOME?.trim() || join(homedir(), '.dsh')).replace(/^~(?=[/\\]|$)/, homedir())),
          REFACTORY_MODEL_PROVIDER: modelConfig?.provider ?? '',
          REFACTORY_MODEL_ID: modelConfig?.model ?? '',
        }
        const runData = typeof payload.dataCommand === 'string' && payload.dataCommand !== ''
          ? payload.dataCommand
          : config.dataCommand
        const runAgent = typeof payload.agentCommand === 'string' ? payload.agentCommand : (config.agentCommand ?? '')
        openedBySession.set(owner, {
          session: owner,
          spine,
          workdir,
          name: typeof payload.name === 'string' && payload.name !== '' ? payload.name : graph.name,
          // The run's own values when the payload carries them; the composition's
          // static ones otherwise, which is what the graph profile relies on.
          commandEnv: runEnv,
          dataCommand: runData,
          agentCommand: runAgent,
          gateCommand: typeof payload.gateCommand === 'string' ? payload.gateCommand : (config.gateCommand ?? ''),
        })
        // A fresh spine has no materialised item. Keep the prior run's files
        // intact; its terminal claim is replaced only after this open succeeds.
        // The post-drain view, not `spine.next()`: a deterministic start node runs
        // here, exactly as the next `<toolPrefix>_next` would, so it is never
        // reported as work the model owes.
        return report(spine, workdir, runEnv, runData, runAgent)
      },
    }))
  }

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_next`,
    description: 'Get the next node(s) the workflow graph says are due now. Deterministic nodes (a command an `fn` gate or an `FnNode` runs) and a data node\'s item passes are executed for you and do not appear, except that when a mode is open interactively one dispatch boundary runs per call: every deterministic node the cursor holds at once is a fork peer the graph declared concurrent, so they are dispatched together and a node that ran is named under `ran` (with its id, declared outputs, a facts summary — `{ _summary, path, keys, sizes, inline }` with values ≤ 512 characters inlined and larger ones reported by key and size only — `peers` naming the whole boundary when more than one ran, and, for a gate the runtime evaluated, its `verdict` and the `reason` it gave — when that sent the run back, `feedback` carries the same reason to the node it went back to, so tell the person why), and `nodes` then names what is due next — which may itself be deterministic, in which case call this tool again rather than running it or calling `wf_done`. Returns `{ status, nodes }`: for `ready`, the node views you must act on (agent task / one-shot LLM / user gate / fork) with the fields you need, plus `item` while a data node is iterating; for `done`, the run is complete; for `failed`, a deterministic node failed (`node` is the node id, `command` is its command string, `error` contains the `<id> exited with code N` prefix followed by the reason); its fork peers that did run have completed, and the failure names the node that did not. A top-level `warnings` lists run warnings that steps this call ran printed and no `ran` step carries, each led by its node id: tell the person what each one says. With automatic execution, sole static ForkNode/JoinNode boundaries are crossed for you and returned branch prompts already have their delivery contexts. Dispatch only returned AgentNode/LLMNode prompts, never fork target ids from a state view. Pass each returned prompt in full; coordination notes must not weaken its requirements or waive the original brief/approved design without explicit user steering. A documented limitation is not a waiver. If a control node is explicitly returned (manual execution or a mixed cursor), acknowledge it with `wf_done`, then call this tool before dispatching branches. Always call this first and act only on what it returns — never skip or reorder.',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('next', value),
    },
    presentCall: args => presentCall('next', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('next', args as Record<string, unknown>, result),
    execute(_args, exec) {
      const { spine, workdir, commandEnv, dataCommand, agentCommand } = requireRun(exec)
      return report(spine, workdir, commandEnv, dataCommand, agentCommand)
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_done`,
    description: 'Mark one non-gate node complete and advance the spine. Returns only a confirmation — call `wf_next` to learn the next node. Static fork/join boundaries normally advance through `wf_next`. Only if it explicitly returns a due ForkNode/JoinNode, acknowledge that control here and call `wf_next` before dispatching branches. For an AgentNode/LLMNode without a dispatch command, call this only after the subagent you started finishes. Never call this for a deterministic node (`FnNode`/`Study`, or an `fn` gate) when the composition runs commands: the runtime runs those itself on the next `wf_next`, and marking one complete here would skip its work.',
    parameters: {
      node: { type: 'string', required: true, description: 'The node id to mark complete.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('done', value),
    },
    presentCall: args => presentCall('done', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('done', args as Record<string, unknown>, result),
    execute(args, exec) {
      const { spine, workdir, commandEnv, agentCommand } = requireRun(exec)
      spine.assertDue(args.node)
      const node = spine.view(args.node)
      if (config.executeCommands && (isDeterministic(node) || node._type === 'DataNode')) {
        throw new Error(`${args.node} is executed by the runtime; call ${toolPrefix}_next instead of declaring it complete`)
      }
      // C2: For review/audit nodes, validate that the CEO's recorded prompt
      // contains the pending delivery's prompt.txt (#347, C2).
      const runKey = workdir ?? '_configured_'
      if (agentCommand !== '' && workdir !== undefined && (node._type === 'AgentNode' || node._type === 'LLMNode')) {
        // Lazily fetch the reviewer list from the runtime bridge.
        if (reviewerIds === undefined) {
          try {
            const rv = runtimeBridge(agentCommand, workdir, commandEnv, { action: 'reviewers' })
            const list = rv.reviewers
            reviewerIds = new Set(Array.isArray(list) ? list.map(String) : [])
          } catch {
            // Fail closed: if we cannot read the reviewer list, treat all nodes as reviewers
            reviewerIds = null
          }
        }
        // Validate prompt for reviewer nodes
        let sentPromptHash: string | undefined
        let sentPromptText: string | undefined
        if (reviewerIds === null || reviewerIds.has(args.node)) {
          const mapKey = `${runKey}:${args.node}`
          // Fail closed: re-read prompt.txt from the pending delivery when
          // the in-memory cache is empty (e.g. after a restart).
          let pending = pendingPrompts.get(mapKey)
          if (pending === undefined) {
            try {
              const pp = runtimeBridge(agentCommand, workdir, commandEnv, { action: 'pending-prompt', node: args.node })
              if (typeof pp.prompt === 'string' && pp.prompt) {
                pending = pp.prompt
                pendingPrompts.set(mapKey, pending)
              }
            } catch { /* bridge failed — pending stays undefined, will refuse below */ }
          }
          if (pending !== undefined) {
            // Check only the latest prompt recorded for this node's current
            // delivery, so a rewritten second dispatch is refused (#347, C2).
            const allRecorded = ceoPrompts.get(mapKey)
            if (allRecorded === undefined || allRecorded.length === 0) {
              throw new Error(
                `wf_done refused: no recorded CEO prompt for reviewer node "${args.node}". `
                + 'The prompt the CEO dispatched must include the pending delivery prompt.',
              )
            }
            // The latest recorded prompt for this node must contain the
            // pending prompt.  No fallback to earlier prompts: if the CEO
            // sent a rewritten retry, the latest is the rewritten one.
            const latest = allRecorded[allRecorded.length - 1]!
            if (latest.prompt.includes(pending!)) {
              sentPromptHash = latest.hash
              sentPromptText = latest.prompt
            } else {
              throw new Error(
                `wf_done refused: the latest recorded CEO prompt for reviewer node "${args.node}" does not contain `
                + 'the delivery\'s prompt.txt. The prompt the CEO dispatched must include the pending delivery prompt.',
              )
            }
          } else {
            // No pending prompt available — fail closed
            throw new Error(
              `wf_done refused: no pending delivery prompt for reviewer node "${args.node}". `
              + 'The delivery context must be established before completing a reviewer node.',
            )
          }
        }
        const mapKey = `${runKey}:${args.node}`
        const nodeUnresolved = pendingUnresolvedKeys.get(mapKey)
        runtimeBridge(agentCommand, workdir, commandEnv, {
          action: 'finish', node: args.node,
          ...(sentPromptHash !== undefined ? { sent_prompt_hash: sentPromptHash } : {}),
          ...(sentPromptText !== undefined ? { sent_prompt: sentPromptText } : {}),
          ...(nodeUnresolved !== undefined ? { unresolved_keys: nodeUnresolved } : {}),
        })
      }
      spine.complete(args.node)
      // Clear pending prompt, CEO prompts, and unresolved keys for completed node (#347, C2)
      const completedMapKey = `${runKey}:${args.node}`
      pendingPrompts.delete(completedMapKey)
      pendingUnresolvedKeys.delete(completedMapKey)
      unresolvedKeyWarned.delete(completedMapKey)
      ceoPrompts.delete(completedMapKey)
      return sync(json({ status: 'complete', node: args.node }))
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_verdict`,
    description: 'Route a currently due gate node. When the runtime executes commands, an automatic gate must run via `wf_next`; the only manual outcome is literal `halt` on a gate with no forward halt edge, which cancels the run without certifying a result. Otherwise, pass one of the `verdicts` the gate view from `wf_next` lists: `proceed` advances along the proceed edge, `reloop` rewinds to the gate\'s reloop target with `feedback` (bounded by the iteration cap), `halt` stops or follows the halt edge, and other labels are forward unless a residual single-gate cycle identifies them as retries under the same cap. A `user` gate is not yours to decide: call this tool with the gate id **and no `verdict`** — the runtime asks the person, routes the answer they give, and refuses when nobody can be asked. There is deliberately no way to answer for them. Returns only a confirmation — call `wf_next` to learn the next node.',
    parameters: {
      gate: { type: 'string', required: true, description: 'The gate node id.' },
      verdict: { type: 'string', description: 'For a manually driven gate: one of its listed `verdicts`. Automatic gates use `wf_next`; literal `halt` without a forward halt edge cancels. Omit it for a `user` gate — the runtime puts that gate to the person, and a verdict passed here is refused rather than recorded, because a person\'s answer is not the caller\'s to give.' },
      feedback: { type: 'string', description: 'Feedback to attach to the reloop target (reloop only).' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('verdict', value),
    },
    presentCall: args => presentCall('verdict', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('verdict', args as Record<string, unknown>, result),
    async execute(args, exec) {
      const { spine, workdir } = requireRun(exec)
      spine.assertDue(args.gate)
      const view = spine.view(args.gate)
      if (config.executeCommands && isDeterministic(view)) {
        if (args.verdict === 'halt' && !spine.acceptedVerdicts(args.gate).includes('halt')) {
          spine.cancel(args.gate)
          recordStatus(spine, workdir, 'halted')
          return json({ status: 'cancelled', gate: args.gate })
        }
        throw new Error(`${args.gate} is evaluated by the runtime; call ${toolPrefix}_next instead of supplying its verdict`)
      }
      if (view.evaluator_type === 'user') {
        if (args.verdict !== undefined) {
          throw new Error(`gate ${args.gate} is a user gate: calling it with a verdict would record an answer the person never gave. Call again with the gate id alone; the runtime asks them.`)
        }
        const result = await askPerson(args.gate, spine, view, exec, workdir)
        if (result === undefined) {
          const status = spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : 'ready'
          recordStatus(spine, workdir, status)
          const state = spine.state()
          return json({ status, gate: args.gate, interrupted_by: 'graph_control',
            nodes: state.cursor, completed: state.completed,
            ...(state.halted === undefined ? {} : { halted: state.halted }) })
        }
        if ('status' in result) {
          return json({
            status: 'replied', gate: args.gate, text: result.text,
            note: 'The person answered with text that is none of the gate\'s outcomes. The gate was asked correctly and remains due. Answer what they wrote in conversation, then call this tool again with the gate alone straight away, so the question is shown again under your reply. Do not ask them to type an outcome in chat, ask the gate\'s question yourself, or recommend an outcome.',
          })
        }
        spine.verdict(args.gate, result.verdict, result.feedback || args.feedback || '')
        if (spine.isHalted()) recordStatus(spine, workdir, 'halted')
        const limit = limitMessage(spine, workdir)
        return json({
          status: 'routed', gate: args.gate, verdict: result.verdict, answered_by: 'user',
          ...(result.feedback ? { feedback: result.feedback } : {}),
          ...(spine.isHalted() ? { halted: true } : {}),
          ...(limit === undefined ? {} : { message: limit }),
        })
      }
      if (args.verdict === undefined) {
        const offered = (view.verdicts ?? []).join(', ')
        throw new Error(`gate ${args.gate} routes on a verdict: pass one of ${offered || 'the verdicts the gate view lists'}`)
      }
      spine.verdict(args.gate, args.verdict, args.feedback ?? '')
      return json({ status: 'routed', gate: args.gate, verdict: args.verdict })
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_edit`,
    description: 'Edit the live graph: `pause` arms a stop BEFORE the named node (including before data setup); `unpause` releases only that boundary after the person asks to continue. A paused run stays stopped across wf_next calls; skip, redo and expansion require explicit unpause first. An armed boundary cannot be skipped. `skip` bypasses a due or future node through its unambiguous continuation, preserving concurrent peers; skipping an unopened fork/data node bypasses its body. Conditional-only gates and already-entered containers require explicit routing or finishing their current work. `unskip` restores a future skipped node. If you already dispatched a worker for skipped work, stop that owned worker explicitly; editing the spine does not cancel it; `set-prompt` replaces an agent/llm node\'s prompt and `append-prompt` adds a paragraph to it; `set-prompt` is refused when the replacement drops a template placeholder (a double-brace variable) present in the current prompt — use `append-prompt` instead. This is how a person\'s correction reaches the node\'s prompt — follow it with `wf_redo` to run that node again. Returns only a confirmation — call `wf_next` to learn the next node. Do not use skip on a check node (a node that produces runtime evidence); only a person can skip the check via graph-control.',
    parameters: {
      node: { type: 'string', required: true, description: 'The node id to edit.' },
      action: { type: 'string', required: true, enum: ['skip', 'unskip', 'set-prompt', 'append-prompt', 'pause', 'unpause'], description: 'The edit.' },
      text: { type: 'string', description: 'The new prompt text; required for `set-prompt` and `append-prompt`.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('edit', value),
    },
    presentCall: args => presentCall('edit', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('edit', args as Record<string, unknown>, result),
    execute(args, exec) {
      const { spine, workdir, commandEnv, agentCommand } = requireRun(exec)
      if (args.action === 'pause' || args.action === 'unpause') {
        spine.pauseBefore(args.node, args.action === 'pause')
        recordStatus(spine, workdir, spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : spine.next().status)
      } else if (args.action === 'skip') {
        if (recordedStatus(workdir) === 'failed') {
          throw new Error('run has failed; a CEO redo cannot spend more attempts. Use '
            + '`refactory graph-control rerun --acknowledge-failure` to have a person spend more attempts.')
        }
        const nodes = spine.planSkip(args.node)
        if (agentCommand !== '' && workdir !== undefined) {
          runtimeBridge(agentCommand, workdir, commandEnv, {
            action: 'invalidate', nodes, reason: `skip ${args.node}`,
          })
        }
        spine.skip(args.node, { source: 'ceo', at: new Date().toISOString() })
      } else if (args.action === 'unskip') spine.unskip(args.node)
      else {
        if (args.text === undefined || args.text.trim() === '') {
          throw new Error(`${toolPrefix}_edit action '${args.action}' needs a non-empty prompt amendment in 'text'`)
        }
        spine.amendPrompt(args.node, args.text, args.action === 'set-prompt' ? 'set' : 'append')
      }
      return sync(json({ status: 'edited', node: args.node, action: args.action }))
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_redo`,
    description: 'Redo completed work after a correction, invalidating downstream completions and produced facts while archiving prior facts. Within an active data item it keeps that item; after a dataset finished, redoing a body node restarts the whole dataset. Concurrent forks must finish first. Returns the restart and invalidated nodes; call `wf_next` to resume.',
    parameters: {
      node: { type: 'string', required: true, description: 'The completed node id to reopen.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('redo', value),
    },
    presentCall: args => presentCall('redo', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('redo', args as Record<string, unknown>, result),
    execute(args, exec) {
      const { spine, workdir, commandEnv, dataCommand } = requireRun(exec)
      // A limit that stopped the run -- a gate halted at its iteration limit, or the run
      // failed, as design mode's own attempt limit fails it -- is not the CEO's to lift.
      const state = spine.state()
      const halted = state.halted as { gate?: string; reason?: string } | undefined
      const stopped = halted?.reason === 'max_iterations'
        ? `run halted at gate ${halted.gate} after reaching its iteration limit`
        : recordedStatus(workdir) === 'failed' ? 'run has failed' : undefined
      if (stopped !== undefined) {
        throw new Error(`${stopped}; a CEO redo cannot spend more attempts. Use `
          + '`refactory graph-control rerun --acknowledge-failure` to have a person spend more attempts.')
      }
      const plan = spine.planRedo(args.node)
      if (dataCommand !== '' && workdir === undefined) {
        throw new Error('redo needs the run workdir to invalidate its data facts safely')
      }
      // Bookkeeping must succeed before the live cursor is changed. This bridge
      // uses the existing data loader because it owns the run's Task facts.
      const receipt = dataCommand !== '' && workdir !== undefined
        ? runtimeBridge(`${dataCommand} reset`, workdir, commandEnv, { ...plan, state, preserveAttemptFacts: true })
        : undefined
      spine.redo(args.node, { preserveIterations: true })
      if (plan.resetPass) materialised.delete(spine)
      recordStatus(spine, workdir, spine.isPaused() ? 'paused' : 'ready')
      return sync(json({ status: 'redone', node: args.node, ...plan, ...(receipt === undefined ? {} : { receipt }) }))
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_expand`,
    description: 'Hot-plug a director\'s N child agents from its plan entries. Returns the child ids it created. Call `wf_next` afterward to learn the next node(s) due. Use this after a director agent produced its plan: pass each `{ slug, prompt }` entry so the spine fans out and runs them.',
    parameters: {
      director: { type: 'string', required: true, description: 'The director node id.' },
      entries: { type: 'array', required: true, items: { type: 'object', additionalProperties: true }, description: 'The plan entries (each with slug + prompt fields).' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('expand', value),
    },
    presentCall: args => presentCall('expand', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('expand', args as Record<string, unknown>, result),
    execute(args, exec) {
      const { spine } = requireRun(exec)
      const entries = args.entries as Array<Record<string, unknown>>
      const result = spine.expand(args.director, entries)
      return sync(json({ status: 'expanded', director: result.expanded, children: result.children }))
    },
  }))

  ctx.tools.register(defineTool({
    name: `${toolPrefix}_state`,
    description: 'Return the full live state: cursor, completed, skipped, gate verdicts, iteration counts, feedback, amended prompts, and the run log.',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [text(value)],
      presentationMeta: (_args, value) => presentationMeta('state', value),
    },
    presentCall: args => presentCall('state', args as Record<string, unknown>),
    presentResult: (args, result) => presentResult('state', args as Record<string, unknown>, result),
    execute(_args, exec) {
      const { spine } = requireRun(exec)
      return sync(json(spine.state()))
    },
  }))

  // The live steering channel: a person's mid-run correction reaching the driver,
  // not only the next agent node. The ledger's owner is the configured command, so
  // the spine learns only a command to run and returns the JSON it prints
  // unchanged — it never parses the ledger itself. Registered when a command is
  // configured and there will be a run directory to read: the configured one, or
  // the one a session opens at run time. Without either there is nothing to read,
  // and a tool that cannot run must not be advertised as if it could.
  if (config.steerCommand !== '' && (config.workdir !== undefined || interactiveOpen)) {
    const command = config.steerCommand
    const steerArgs = [...config.steerArgs]
    ctx.tools.register(defineTool({
      name: `${toolPrefix}_steer`,
      description: `Read the run’s intent ledger, or append a note with optional kind (info, correction, constraint) and source. A write returns the updated ledger; an invalid or failed write is an error, never a read-only success. Read what was said about this run since you last looked, before deciding anything. Runs the run's steering command and returns its ledger payload unchanged: the origin, every entry (a note, a correction, a constraint), recorded gate verdicts, and the fidelity problems found. Read it before each \`${toolPrefix}_next\`; when it carries a correction, apply it to that node's prompt with \`${toolPrefix}_edit\` (\`set-prompt\`/\`append-prompt\`; prefer \`append-prompt\` for nodes with template placeholders) and re-run the node with \`${toolPrefix}_redo\`. On a read, an unreadable or unavailable ledger is reported as \`status: 'unavailable'\`, never thrown, so it cannot stop the run.${effectivePersonAttached ? ` When relaying the person's own words, set source to 'user-relayed' so they become auditable requirements; your own observations default to 'ceo' and are context only.` : ''}`,
      parameters: {
        note: { type: 'string', description: 'Append a note, correction, or constraint to this run. Omit to read without writing.' },
        kind: { type: 'string', enum: ['info', 'correction', 'constraint'], description: 'The note kind; defaults to info. Requires note.' },
        source: { type: 'string', enum: ['user-relayed', 'ceo'], description: "Who said this: 'user-relayed' for the person's words you are relaying (they become auditable requirements), 'ceo' (default) for your own observations (context only, not auditable). When in doubt, use 'ceo'; mark as 'user-relayed' only what the person actually said." },
      },
      output: {
        schema: { type: 'json' },
        render: (_args, value) => [text(value)],
        presentationMeta: (_args, value) => presentationMeta('steer', value),
      },
      presentCall: args => presentCall('steer', args as Record<string, unknown>),
      presentResult: (args, result) => presentResult('steer', args as Record<string, unknown>, result),
      execute(args, exec) {
        const unknown = Object.keys(args).filter(key => key !== 'note' && key !== 'kind' && key !== 'source')
        if (unknown.length > 0) throw new Error(`${toolPrefix}_steer does not support: ${unknown.join(', ')}`)
        if (args.kind !== undefined && args.note === undefined) throw new Error('kind requires a note')
        if (args.note !== undefined && args.note.trim() === '') throw new Error('an empty note records nothing')
        // The ledger lives in the run directory, which for an opened run is the
        // one its `wf_open` payload named — not the composition's configured one.
        const run = requireRun(exec)
        if (run.workdir === undefined) {
          if (args.note !== undefined) throw new Error('this run has no work directory to record a note')
          return sync(json({ status: 'unavailable', error: 'this run has no work directory to read' }))
        }
        const argv = [...steerArgs, run.workdir]
        let effectiveSource = args.source ?? 'ceo'
        if (effectiveSource === 'user-relayed' && !effectivePersonAttached) {
          effectiveSource = 'ceo'
        }
        if (args.note !== undefined) argv.push(`--note=${args.note}`, `--kind=${args.kind ?? 'info'}`, `--source=${effectiveSource}`)
        const result = spawnSync(command, argv, { encoding: 'utf8' })
        if (result.status === 0) {
          try {
            return sync(json(JSON.parse(result.stdout)))
          } catch {
            // Exit 0 with unparseable output is as unreadable as a failure: fall
            // through to the unavailable payload instead of throwing.
          }
        }
        const reason = result.error?.message || lastLine(result.stderr) || lastLine(result.stdout) || 'no output'
        if (args.note !== undefined) throw new Error(`${toolPrefix}_steer could not record the note: ${reason}`)
        return sync(json({ status: 'unavailable', error: reason }))
      },
    }))
  }

  // Populate childToParent for write-confinement (#347, C0). Registered before
  // the CEO deny guard so the map is ready before any restriction is applied.
  ctx.on('agent/created', ({ agent }) => {
    const parentId = agent.session.header.parentSession
    if (parentId !== undefined) {
      childToParent.set(String(agent.session.id), String(parentId))
    }
  })
  ctx.on('agent/disposed', ({ agent }) => {
    childToParent.delete(String(agent.session.id))
  })

  // The drift-free guard. At `agent/created` the CEO (a top-level agent, never
  // a subagent) of THIS composition loses the coding/external tools on its own
  // scope only. With no composition scope (a host-level mount, as on the CLI
  // graph profile), every top-level agent on the host is the CEO. Content
  // subagents are siblings of the CEO under the same standing mount, so this
  // restriction does not reach them.
  const compositionScope = scopeOf(ctx)
  const restrictions = new WeakMap<Agent, () => void>()
  ctx.on('agent/created', ({ agent }) => {
    if (agent.session.header.origin === 'subagent') return
    const agentScope = scopeOf(agent.ctx)
    if (agentScope === undefined) return
    if (compositionScope !== undefined && !scopeChainOf(agentScope).includes(compositionScope)) return
    const visible = new Set(ctx.tools.schemas(agentScope).map(schema => schema.name))
    const deny = ceoDeny.filter(name => visible.has(name))
    if (deny.length === 0) return
    restrictions.set(agent, agent.ctx.tools.restrict({ deny }))
  })
  ctx.on('agent/disposed', ({ agent }) => {
    const dispose = restrictions.get(agent)
    if (dispose === undefined) return
    dispose()
    restrictions.delete(agent)
  })
}
