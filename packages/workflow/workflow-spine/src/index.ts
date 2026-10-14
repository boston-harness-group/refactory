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
 * happens inside the subagents the CEO spawns, which join the full `standard`
 * composition and keep every tool. Removing the coding tools from the CEO is
 * what turns "please follow the graph" into "there is no other path".
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
import { Spine, validateGraph } from './spine.ts'
import type { NodeView, SpineNext } from './spine.ts'
import { commandOf, isDeterministic, runDataCommand, runDeterministic } from './execute.ts'
import type { CommandFailure } from './execute.ts'
import type { WorkflowGraph } from './types.ts'

export { Spine, validateGraph } from './spine.ts'
export { commandOf, dataCommandOf, isDeterministic, runDataCommand, verdictOf, NODE_ENV_KEY } from './execute.ts'
export type { CommandFailure, CommandOutcome, CommandRunnerConfig, NodeManifest } from './execute.ts'
export type { DataPass, GraphIssue, NodeView, PromptAmendmentMode, SpineNext } from './spine.ts'
export type { WorkflowEdge, WorkflowGraph, WorkflowNode } from './types.ts'

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
   * CEO's own scope at `agent/created`, so subagents (which join the full
   * `standard` composition as siblings) keep them. Names absent from the CEO's
   * inherited surface are skipped, so the list can name platform-gated tools
   * (e.g. `pwsh`) that are not present.
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
 * @returns the node's id and type, plus the outputs and facts that exist.
 */
function recordedStep(node: NodeView, workdir: string | undefined, peers?: NodeView[]): RanStep {
  const step: RanStep = { node: node.id, type: node._type }
  if (peers !== undefined && peers.length > 1) step.peers = peers.map(peer => peer.id)
  if (workdir === undefined) return step
  const outputs = (node.writes ?? []).filter(rel => rel !== '' && existsSync(resolve(workdir, rel)))
  if (outputs.length > 0) step.outputs = outputs
  try {
    // The runner writes one JSON object; anything else is not facts a reader can
    // use, and is reported as absent rather than as a malformed value.
    const parsed: unknown = JSON.parse(readFileSync(join(workdir, 'facts.json'), 'utf8'))
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      step.facts = parsed as JsonValue
    }
  } catch {
    // No facts file yet, or one a command left mid-write: the run directory
    // simply carries no facts for this step.
  }
  return step
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
  /** The node's declared writes that now exist in the run directory, relative to it. */
  outputs?: string[]
  /** The run's `facts.json` as the node's command left it. */
  facts?: JsonValue
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

export function apply(ctx: Context, config: Config): void {
  const { toolPrefix, maxIterations, ceoDeny, interactiveOpen } = config
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

  // Who answers a `user` gate. Resolved here, once, so routing one is a runtime act
  // rather than a claim the model makes: a composition without an answerer (a
  // headless run) cannot route a person's gate at all, and a model cannot route one
  // on their behalf.
  const userQuestions: UserQuestionService | undefined = ctx.get('userQuestions')

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
    if (saved.status === 'executing' || saved.status === 'failed') throw new Error('unknown side-effect outcome; automatic replay refused')
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
   * @param detail - the error, for a failure or refusal.
   */
  function recordStatus(spine: Spine, workdir: string | undefined, status: string, detail?: string): void {
    if (workdir === undefined) return
    try {
      const dir = join(workdir, '.refactory')
      mkdirSync(dir, { recursive: true })
      // The cursor travels with the status: a reader of a finished run wants the
      // score, and a reader of an abandoned one wants to know which nodes never ran.
      const state = spine.state()
      const next = spine.next()
      const views = next.status === 'ready' ? next.nodes : []
      const wait = status === 'paused' ? 'paused by user'
        : status === 'waiting_user' || views.some(node => node._type === 'GateNode' && node.evaluator_type === 'user') ? 'waiting for user gate'
          : views.some(node =>
              (node._type === 'AgentNode' && !(node.command !== undefined && node.command !== ''))
              || node._type === 'LLMNode'
            ) ? 'waiting for worker delivery'
            : status === 'executing' ? 'running a graph check' : undefined
      const record = {
        schema_version: 1, run_id: resolve(workdir), at: new Date().toISOString(),
        status, nodes: state.cursor, completed: state.completed,
        ...(detail === undefined ? {} : { error: detail }),
        ...(wait === undefined ? {} : { wait_reason: wait }),
        ...(status === 'paused' ? { paused_before: spine.pausedAt() } : {}),
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
        file_fingerprints, materialised: materialised.get(spine) ?? null, run_id: resolve(workdir) }))
      renameSync(`${checkpoint}.tmp`, checkpoint)
      appendFileSync(join(dir, 'events.jsonl'), JSON.stringify({ ...record, kind: 'graph_state' }) + '\n')

    } catch (error) {
      // A stale ready checkpoint could replay a command whose outcome is unknown.
      // Persist the execution marker before dispatch, and stop on every write error.
      throw new Error('Cannot persist graph recovery checkpoint; continuation refused', { cause: error })
    }
  }

  /** Consume only controls addressed to this exact durable invocation. */
  function applyControls(spine: Spine, workdir: string | undefined, env: Record<string, string>, dataCommand: string): void {
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
          const cursor = state.cursor as string[]
          const targets = cursor.length ? cursor : spine.passOf() ? [spine.passOf()!.node] : []
          for (const node of targets) spine.pauseBefore(node, true)
        }
        else if (request.action === 'resume') for (const node of state.pause_before as string[]) spine.pauseBefore(node, false)
        else if (request.action === 'stop') { spine.stopAtBoundary(); receipt.message = 'Stopped at graph boundary; no further nodes will dispatch.' }
        else if (request.action === 'rerun') {
          const checkpoint = readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'))
          if (createHash('sha256').update(checkpoint).digest('hex') !== request.revision) throw new Error('graph changed since preview; preview invalidation again')
          const target = String(request.target)
          const plan = spine.planRedo(target)
          const nodes = (spine.checkpoint().graph as WorkflowGraph).nodes
          if (typeof request.text === 'string' && request.text.trim() !== '' && !['AgentNode', 'LLMNode'].includes(nodes[target]?._type ?? '')) {
            throw new Error('this node has no editable prompt; correct its source with explicit invalidation')
          }
          if (dataCommand !== '') {
            recordStatus(spine, workdir, 'executing')
            runtimeBridge(`${dataCommand} reset`, workdir, env, { ...plan, state, archiveArtifacts: true })
          }
          if (typeof request.text === 'string' && request.text.trim() !== '') spine.amendPrompt(target, request.text, 'append')
          spine.redo(target)
          if (plan.resetPass) materialised.delete(spine)
          receipt.message = `Reopened ${target}; invalidated ${plan.nodes.join(', ')}; prior evidence archived.`
        } else throw new Error('unsupported graph control')
      } catch (error) { receipt.status = 'failed'; receipt.message = error instanceof Error ? error.message : String(error) }
      if (receipt.status === 'applied') recordStatus(spine, workdir,
        spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : spine.next().status)
      writeFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), JSON.stringify(receipt))
      appendFileSync(join(workdir, '.refactory', 'events.jsonl'), JSON.stringify({ schema_version: 1,
        run_id: resolve(workdir), invocation, at: new Date().toISOString(), kind: 'control',
        state: receipt.status, summary: receipt.message }) + '\n')
      unlinkSync(path)
    }
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
   */
  async function drain(
    spine: Spine,
    workdir: string | undefined,
    commandEnv: Record<string, string>,
    dataCommand: string,
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
      applyControls(spine, workdir, commandEnv, dataCommand)
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
      applyControls(spine, workdir, commandEnv, dataCommand)
      if (spine.isHalted() || spine.isPaused()) return { next: spine.next(), item, ran }
      const closed = await close()
      if (closed.failure !== undefined) return { next: spine.next(), failure: closed.failure, ran }
      let state = spine.next()
      if (state.status === 'done' || spine.isPaused()) return { next: state, item, ran }
      const started = await beginItem()
      if (started.failure !== undefined) return { next: state, failure: started.failure, ran }
      applyControls(spine, workdir, commandEnv, dataCommand)
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
      const outcomes = await Promise.all(automatic.map(async node => {
        // A finishing command hands its slot straight to the first waiter rather
        // than returning it to the pool, so a fresh starter cannot slip in
        // between the release and the waiter's wakeup and overshoot the ceiling.
        if (inFlight >= ceiling) await new Promise<void>(resolve => waiters.push(resolve))
        else inFlight += 1
        try {
          return await runDeterministic(shell, node, runner)
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
        if (outcome.verdict === undefined) spine.complete(node.id)
        else spine.verdict(node.id, outcome.verdict)
      }
      recordBoundary()
      if (failure !== undefined) return { next: spine.next(), failure, ran }
      if (interactiveOpen) {
        // Preserve the one-boundary budget while allowing structural boundaries
        // and data-pass grading to settle before exposing the next ready work.
        ran = recordedStep(automatic[automatic.length - 1]!, workdir, automatic)
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
    applyControls(spine, workdir, commandEnv, dataCommand)
    let drained: Awaited<ReturnType<typeof drain>>
    try { drained = await drain(spine, workdir, commandEnv, dataCommand) }
    catch (error) {
      recordStatus(spine, workdir, 'failed', error instanceof Error ? error.message : String(error))
      throw error
    }
    const { next, failure, item, ran } = drained
    if (failure !== undefined) {
      // The node's own stderr is why it failed. The exit code alone sends a reader
      // back to re-run the node to find out, so the reason is recorded with it.
      const reason = failure.stderr.trim()
      recordStatus(spine, workdir, 'failed', reason === '' ? failure.error : `${failure.error}: ${reason.slice(-STATUS_DETAIL)}`)
      return json({
        status: 'failed',
        node: failure.command,
        exit_code: failure.exitCode,
        error: failure.error,
        stdout: failure.stdout,
        stderr: failure.stderr,
      })
    }
    if (next.status === 'ready' && !spine.isPaused() && agentCommand !== '' && workdir !== undefined) {
      for (const node of next.nodes) {
        if (node._type !== 'AgentNode' && node._type !== 'LLMNode') continue
        // Dispatched AgentNodes are run by commandOf()/drain(); skip them here
        // to avoid recording a double delivery.
        if (node._type === 'AgentNode' && node.command !== undefined && node.command !== '') continue
        const receipt = runtimeBridge(agentCommand, workdir, commandEnv, {
          action: 'begin', node: node.id, prompt: node.prompt_template ?? '',
          reads: node.reads ?? [], writes: node.writes ?? [],
        })
        if (typeof receipt.prompt !== 'string') throw new Error('agent delivery bridge returned no prompt')
        node.prompt_template = receipt.prompt
      }
    }
    recordStatus(spine, workdir, spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : next.status === 'done' ? 'done' : 'ready')
    // The item being iterated travels with the node views, so the subagent for a
    // pass is handed the item rather than being sent to find it on disk. Under
    // `interactiveOpen` the node this call ran travels the same way, so the CEO
    // can say which one it was and what it left behind.
    const current = spine.isPaused() ? { ...next, status: 'paused' } : next
    const view = item === undefined ? current : { ...current, item }
    return json(ran === undefined ? view : { ...view, ran })
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
   * Result of a user-gate question.
   * - `{ verdict, feedback }` — a recognized verdict, optionally with parsed feedback.
   * - `{ status: 'replied', gate, text }` — free text that does not match any verdict label;
   *   the gate stays open and the caller should surface the text and re-ask.
   * - `undefined` — halted, paused, or state changed during the question.
   */
  type UserGateResult =
    | { verdict: string; feedback: string }
    | { status: 'replied'; gate: string; text: string }
    | undefined

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
    const { gateCommand, commandEnv, dataCommand } = requireRun(exec)
    applyControls(spine, workdir, commandEnv, dataCommand)
    if (spine.isPaused() || spine.isHalted()) return undefined
    const labels = spine.acceptedVerdicts(gate)
    const request = { gate, reads: view.reads ?? [], labels }
    const prior = gateCommand !== '' && workdir !== undefined
      ? runtimeBridge(gateCommand, workdir, commandEnv, { ...request, action: 'begin' })
      : undefined
    if (typeof prior?.verdict === 'string' && prior.verdict !== '') return { verdict: prior.verdict, feedback: '' }
    const stateBefore = JSON.stringify(spine.state())
    if (userQuestions === undefined) {
      const message = `gate ${gate} is answered by a person and this composition mounts no user-questions `
        + 'channel, so it cannot be routed here; run it where a person can be asked'
      recordStatus(spine, workdir, 'refused', message)
      throw new Error(message)
    }
    let answer
    const questionLifetime = new AbortController()
    const questionSignal = exec.signal === undefined ? questionLifetime.signal : AbortSignal.any([exec.signal, questionLifetime.signal])
    try {
      recordStatus(spine, workdir, 'waiting_user')
      const nativeAnswer = userQuestions.ask({
        questions: [{
          id: gate,
          header: 'Workflow gate',
          question: view.gate_prompt || `Gate ${gate}`,
          ...((view.reads ?? []).length > 0 ? { detail: `Gate reads files: ${(view.reads ?? []).join(', ')}` } : {}),
          ...(labels.length === 0 ? {} : { options: labels.map(verdict => ({ label: verdict })) }),
        }],
        ...(exec.agent === undefined ? {} : { agent: exec.agent }),
        signal: questionSignal,
      })
      const panelAnswer = async (): Promise<Awaited<typeof nativeAnswer> | undefined> => {
        if (workdir === undefined) return new Promise(() => {})
        while (!questionSignal.aborted) {
          await wait(500, undefined, { signal: questionSignal })
          applyControls(spine, workdir, commandEnv, dataCommand)
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
      const message = `gate ${gate} is answered by a person and could not be asked: ${detail}`
      recordStatus(spine, workdir, 'refused', message)
      throw new Error(message)
    } finally { questionLifetime.abort() }
    if (answer === undefined) return undefined
    const chosen = answer.answers.find(entry => entry.id === gate)
    const rawText = (chosen?.selected[0] ?? chosen?.custom ?? '').trim()
    if (rawText === '') {
      throw new Error(`gate ${gate} was answered without a verdict`)
    }
    if (JSON.stringify(spine.state()) !== stateBefore) {
      throw new Error(`gate ${gate} changed while awaiting the answer; ask again`)
    }

    // Detect whether this is a custom-text answer (user typed in "Other" box)
    const isCustomText = (chosen?.selected.length === 0 || chosen?.selected[0] === undefined)
      && typeof chosen?.custom === 'string' && chosen.custom.trim() !== ''

    let verdict: string
    let feedback = ''

    if (!isCustomText) {
      // Normal selection path — enforce label validation as before
      if (labels.length > 0 && !labels.includes(rawText)) {
        throw new Error(`gate ${gate} does not accept ${rawText}`)
      }
      verdict = rawText
    } else if (labels.length === 0) {
      // No labels constrain the gate; accept any text as the verdict
      verdict = rawText
    } else {
      // Case-insensitive exact match, then "label: feedback" prefix match
      const exactMatch = labels.find(l => l.toLowerCase() === rawText.toLowerCase())
      if (exactMatch !== undefined) {
        verdict = exactMatch
      } else {
        // Check for "label: feedback" pattern (longest label first to avoid prefix ambiguity)
        const sortedLabels = [...labels].sort((a, b) => b.length - a.length)
        const matchedLabel = sortedLabels.find(label =>
          rawText.toLowerCase().startsWith(label.toLowerCase() + ':'),
        )
        if (matchedLabel === undefined) {
          // Unrecognized free text → gate stays open, return non-error reply
          recordStatus(spine, workdir, 'ready')
          return { status: 'replied', gate, text: rawText }
        }
        verdict = matchedLabel
        feedback = rawText.slice(matchedLabel.length + 1).trim()
      }
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
      + `directory, so say what it produced before acting on what is due next. Open any mode this way, `
      + 'including an authoring one (create/edit/design): it is the route where a person is present, and '
      + 'a gate marked `(user)` is put to them. For create/edit/design, include the full user brief as '
      + '`brief=<full text>` in the initial knobs array, alongside name/task/check knobs the mode declares. '
      + 'Intent records the same purpose in the authoritative ledger delivered to every agent node; '
      + 'an empty brief knob alone is not a reason to reopen a live run or edit a deterministic FnNode prompt. '
      + 'Use refactory action `runtime` when available to inspect inherited model/home context before asking setup questions. '
      + `A FnNode, Study, \`fn\` gate, or AgentNode with a dispatch command listed in \`nodes\` is deterministic`
      + ' — do NOT run its command and do NOT call '
      + `${toolPrefix}_done for it; call ${toolPrefix}_next again and the runtime runs it. `
      + `${toolPrefix}_done is only for a node whose work you or a subagent did.`
    : ''
  ctx.systemPrompt.section({
    name: `tool:${toolPrefix}-spine`,
    order: ctx.systemPrompt.getSectionOrder('TOOL_WORKFLOW'),
    text: `You are the CEO of a workflow run, and you are DRIVING A WORKFLOW GRAPH. Your coding tools (bash, write, edit, web, skill) are disabled — you cannot implement, edit files, or browse yourself. Advance only through the wf tools.${openInstruction} If the person requests a pause before a named node or deterministic dataset, arm it FIRST with ${toolPrefix}_edit action \`pause\` for that node, before any ${toolPrefix}_next that could reach it. Use ${toolPrefix}_state to inspect ids without executing work. When status is \`paused\`, end your turn and wait for the person. Only after an explicit request to continue, use ${toolPrefix}_edit action \`unpause\` on that boundary and resume. Repeated ${toolPrefix}_next calls do not release a pause; never skip or redo around it. ${toolPrefix}_state inspects the cursor without execution; ${toolPrefix}_next executes due deterministic work. After reading steering and arming any requested pause, call it to advance, and call it again after every ${toolPrefix}_done / ${toolPrefix}_verdict / ${toolPrefix}_edit / ${toolPrefix}_expand — those tools only confirm the mutation, they do not tell you what is next. When ${toolPrefix}_steer is available, read it before each ${toolPrefix}_next: it is what the person said about the run since you last looked. A correction from the person is applied to that node's prompt with ${toolPrefix}_edit (\`set-prompt\`/\`append-prompt\`) and the node is re-run with ${toolPrefix}_redo. With automatic execution enabled, ${toolPrefix}_next crosses sole static ForkNode/JoinNode boundaries for you and establishes delivery contexts before returning the branch AgentNode/LLMNode prompts. Dispatch only those returned prompts in parallel; target ids in a state view are not dispatch instructions. If a ForkNode or JoinNode is explicitly returned (manual execution or a mixed cursor), acknowledge only that due control node with ${toolPrefix}_done, then call ${toolPrefix}_next before dispatching branch work. For an AgentNode or LLMNode (unless the runtime runs it — it has a dispatch command), spawn a subagent (the subagent tool) whose task is the node's prompt_template, and explicitly tell the subagent to read the node's reads files and write its result to the node's writes file (the filename in the node's writes field, e.g. candidate.py) — wait for it, then report the node with ${toolPrefix}_done. Every path in a node's reads and writes is relative to ${runDirectory}, and the subagent's own working directory is NOT that directory: give the subagent the absolute path (${runDirectory}/<the path>) for each file it must read or write, and tell it to write nowhere else. For a FnNode with a command, call ${toolPrefix}_next to execute it; only a command-less FnNode is yours to complete. Automatic gates also run through ${toolPrefix}_next; never supply their evaluator result. For a manually driven gate, use ${toolPrefix}_verdict with one of its listed outcomes. For a user gate the runtime asks the person itself and routes the answer they give: call ${toolPrefix}_verdict with the gate id and no verdict, and never answer for them — a verdict passed for their gate is refused. Before routing a user gate, read and summarise its \`reads\` files (listed in the node view returned by ${toolPrefix}_next) for the person — key decisions, scope, notable risks or non-goals — and link each file, so they see what they are deciding on before the question is put to them. Fan out a director's plan with ${toolPrefix}_expand, and apply user steering ("skip the tests") with ${toolPrefix}_edit. When a person says a step is wrong, read the run's ledger if one is configured, then use ${toolPrefix}_edit with \`set-prompt\` or \`append-prompt\` to correct that node's prompt and ${toolPrefix}_redo to run the node again with the correction. A DataNode is iterated for you: the runtime selects its items and hands you the subgraph's nodes one pass at a time, with the item being processed in the payload's item field and on disk as the node's current item file. Do not try to iterate it yourself, and do not report the data node — or the pass's last node — before its work is done. Never skip, reorder, or bypass a node — the runner refuses to advance until you report the current node complete.`,
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
        knobs: { type: 'array', items: { type: 'string' }, description: 'Knob overrides for this run, each as `NAME=VALUE` (repeatable). For create/edit/design, include `brief=<the full user brief>` on the first open, plus the other declared knobs such as name, task_ref or check.' },
        intent: { type: 'string', description: 'The user’s purpose for this run, recorded in its intent ledger. Without one the ledger records the requested mode identifier.' },
        workdir: { type: 'string', description: 'The run directory to lower into. When absent the spine derives one from this session and the mode — `<cwd>/.factory/web/<session>/<package>-<mode>` — creates it, and passes it as `--workdir`; the console\'s `open` verb requires a workdir, so one is always sent.' },
      },
      output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
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
        const argv = [...config.openArgs, args.mode]
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
    description: 'Get the next node(s) the workflow graph says are due now. Deterministic nodes (a command an `fn` gate or an `FnNode` runs) and a data node\'s item passes are executed for you and do not appear, except that when a mode is open interactively one dispatch boundary runs per call: every deterministic node the cursor holds at once is a fork peer the graph declared concurrent, so they are dispatched together and a node that ran is named under `ran` (with its id, declared outputs, the run\'s facts, and `peers` naming the whole boundary when more than one ran), and `nodes` then names what is due next — which may itself be deterministic, in which case call this tool again rather than running it or calling `wf_done`. Returns `{ status, nodes }`: for `ready`, the node views you must act on (agent task / one-shot LLM / user gate / fork) with the fields you need, plus `item` while a data node is iterating; for `done`, the run is complete; for `failed`, a deterministic node failed; its fork peers that did run have completed, and the failure names the node that did not. With automatic execution, sole static ForkNode/JoinNode boundaries are crossed for you and returned branch prompts already have their delivery contexts. Dispatch only returned AgentNode/LLMNode prompts, never fork target ids from a state view. Pass each returned prompt in full; coordination notes must not weaken its requirements or waive the original brief/approved design without explicit user steering. A documented limitation is not a waiver. If a control node is explicitly returned (manual execution or a mixed cursor), acknowledge it with `wf_done`, then call this tool before dispatching branches. Always call this first and act only on what it returns — never skip or reorder.',
    parameters: {},
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
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
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
    execute(args, exec) {
      const { spine, workdir, commandEnv, agentCommand } = requireRun(exec)
      spine.assertDue(args.node)
      const node = spine.view(args.node)
      if (config.executeCommands && (isDeterministic(node) || node._type === 'DataNode')) {
        throw new Error(`${args.node} is executed by the runtime; call ${toolPrefix}_next instead of declaring it complete`)
      }
      if (agentCommand !== '' && workdir !== undefined && (node._type === 'AgentNode' || node._type === 'LLMNode')) {
        runtimeBridge(agentCommand, workdir, commandEnv, { action: 'finish', node: args.node })
      }
      spine.complete(args.node)
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
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
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
          // Unrecognized free text: gate stays open, return non-error JSON to the CEO
          return json({
            status: 'replied', gate: args.gate, text: result.text,
            note: 'The user typed free text that does not match any accepted verdict. The gate remains due. Respond to the user and call this tool again to re-ask.',
          })
        }
        // Normal verdict path
        spine.verdict(args.gate, result.verdict, result.feedback || args.feedback || '')
        return json({
          status: 'routed', gate: args.gate, verdict: result.verdict, answered_by: 'user',
          ...(result.feedback ? { feedback: result.feedback } : {}),
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
    description: 'Edit the live graph: `pause` arms a stop BEFORE the named node (including before data setup); `unpause` releases only that boundary after the person asks to continue. A paused run stays stopped across wf_next calls; skip, redo and expansion require explicit unpause first. An armed boundary cannot be skipped. `skip` bypasses a due or future node through its unambiguous continuation, preserving concurrent peers; skipping an unopened fork/data node bypasses its body. Conditional-only gates and already-entered containers require explicit routing or finishing their current work. `unskip` restores a future skipped node. If you already dispatched a worker for skipped work, stop that owned worker explicitly; editing the spine does not cancel it; `set-prompt` replaces an agent/llm node\'s prompt and `append-prompt` adds a paragraph to it. This is how a person\'s correction reaches the node\'s prompt — follow it with `wf_redo` to run that node again. Returns only a confirmation — call `wf_next` to learn the next node.',
    parameters: {
      node: { type: 'string', required: true, description: 'The node id to edit.' },
      action: { type: 'string', required: true, enum: ['skip', 'unskip', 'set-prompt', 'append-prompt', 'pause', 'unpause'], description: 'The edit.' },
      text: { type: 'string', description: 'The new prompt text; required for `set-prompt` and `append-prompt`.' },
    },
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
    execute(args, exec) {
      const { spine, workdir, commandEnv, agentCommand } = requireRun(exec)
      if (args.action === 'pause' || args.action === 'unpause') {
        spine.pauseBefore(args.node, args.action === 'pause')
        recordStatus(spine, workdir, spine.isHalted() ? 'halted' : spine.isPaused() ? 'paused' : spine.next().status)
      } else if (args.action === 'skip') {
        const nodes = spine.planSkip(args.node)
        if (agentCommand !== '' && workdir !== undefined) {
          runtimeBridge(agentCommand, workdir, commandEnv, {
            action: 'invalidate', nodes, reason: `skip ${args.node}`,
          })
        }
        spine.skip(args.node)
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
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
    execute(args, exec) {
      const { spine, workdir, commandEnv, dataCommand } = requireRun(exec)
      const plan = spine.planRedo(args.node)
      if (dataCommand !== '' && workdir === undefined) {
        throw new Error('redo needs the run workdir to invalidate its data facts safely')
      }
      // Bookkeeping must succeed before the live cursor is changed. This bridge
      // uses the existing data loader because it owns the run's Task facts.
      const receipt = dataCommand !== '' && workdir !== undefined
        ? runtimeBridge(`${dataCommand} reset`, workdir, commandEnv, { ...plan, state: spine.state() })
        : undefined
      spine.redo(args.node)
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
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
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
    output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
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
      description: `Read the run’s intent ledger, or append the user’s stated note with optional kind (info, correction, constraint). A write returns the updated ledger; an invalid or failed write is an error, never a read-only success. Read what the person said about this run since you last looked, before deciding anything. Runs the run's steering command and returns its ledger payload unchanged: the origin, every entry (a note, a correction, a constraint), recorded gate verdicts, and the fidelity problems found. Read it before each \`${toolPrefix}_next\`; when it carries a correction, apply it to that node's prompt with \`${toolPrefix}_edit\` (\`set-prompt\`/\`append-prompt\`) and re-run the node with \`${toolPrefix}_redo\`. On a read, an unreadable or unavailable ledger is reported as \`status: 'unavailable'\`, never thrown, so it cannot stop the run.`,
      parameters: {
        note: { type: 'string', description: 'Append the user’s stated note, correction, or constraint to this run. Omit to read without writing.' },
        kind: { type: 'string', enum: ['info', 'correction', 'constraint'], description: 'The note kind; defaults to info. Requires note.' },
      },
      output: { schema: { type: 'json' }, render: (_args, value) => [text(value)] },
      execute(args, exec) {
        const unknown = Object.keys(args).filter(key => key !== 'note' && key !== 'kind')
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
        if (args.note !== undefined) argv.push(`--note=${args.note}`, `--kind=${args.kind ?? 'info'}`)
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

  // The drift-free guard. At `agent/created` the CEO (a top-level agent, never
  // a subagent) of THIS composition loses the coding/external tools on its own
  // scope only. Content subagents are siblings of the CEO under the same
  // standing mount, so this restriction does not reach them.
  const compositionScope = scopeOf(ctx)
  const restrictions = new WeakMap<Agent, () => void>()
  ctx.on('agent/created', ({ agent }) => {
    if (agent.session.header.origin === 'subagent') return
    if (compositionScope === undefined) return
    const agentScope = scopeOf(agent.ctx)
    if (agentScope === undefined) return
    if (!scopeChainOf(agentScope).includes(compositionScope)) return
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
