/**
 * Running a graph's deterministic nodes against a shell executor.
 *
 * An `FnNode` with a `command` and an `fn` `GateNode` with an `evaluator_command`
 * need no model: the plugin can run them itself, so `wf_next` hands the driving
 * agent only the nodes that need one. Keeping this out of `spine.ts` leaves the
 * cursor state machine free of I/O.
 * @module @deepseek-ai/dsh-workflow-spine/execute
 */

import { spawnSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { ShellExecRequest, ShellExecSpec, ShellExecutor, ShellRunResult } from '@deepseek-ai/dsh-shell'
import type { NodeView } from './spine.ts'

/**
 * Environment key carrying the running node's declaration as JSON. A node
 * command is generic — it knows nothing about the graph that invoked it — so
 * the plugin hands it the declaration it needs (its id and the artifacts it
 * reads and writes) under one harness-owned name.
 */
export const NODE_ENV_KEY = 'DSH_WORKFLOW_NODE'

/** What a node command is told about the node it is running as. */
export interface NodeManifest {
  id: string
  type: string
  command: string
  reads: string[]
  writes: string[]
  prompt_file?: string
  feedback_file?: string
}

/** Per-run settings for node commands: where they run, and with what. */
export interface CommandRunnerConfig {
  /** Working directory for every node command; absent uses the executor's own. */
  workdir?: string
  /** Extra environment entries merged into each node command. */
  env: Record<string, string>
  /** Foreground timeout for one node command. */
  timeoutMs: number
  /**
   * Per-command sandbox policy. A graph's commands are trusted deterministic
   * ops, not model-authored code, so they run with full access by default; the
   * model-authored candidate is confined by the op that scores it.
   */
  sandboxPolicy?: { mode: 'read-only' | 'workspace-write' | 'danger-full-access'; workspaceRoot: string }
}

/** Check the shared ledger before dispatch and after completion, including final validation. */
function resourceCheck(node: NodeView, config: CommandRunnerConfig): number | undefined {
  const ledger = config.env.REFACTORY_BUDGET_LEDGER ?? process.env.REFACTORY_BUDGET_LEDGER
  if (ledger === undefined) return undefined
  const phase = node.resource_role
  const pythonBin = config.env.REFACTORY_PYTHON_BIN ?? 'python'
  const checked = spawnSync(pythonBin, ['-m', 'refactory_dsh.budgets', ledger, '--check', ...(phase === undefined ? [] : ['--phase', phase])], {
    cwd: config.workdir, env: { ...process.env, ...config.env }, encoding: 'utf8', timeout: 10_000,
  })
  if (checked.status !== 0) throw new Error(`Resource boundary refused: ${checked.stdout || checked.stderr || checked.error?.message}`)
  const state = JSON.parse(checked.stdout) as { elapsed_s: number; phase: string; limits: { max_elapsed_s: number | null; validation_reserve_s: number } }
  if (state.limits.max_elapsed_s === null) return undefined
  const reserve = state.phase === 'final_validation' ? 0 : state.limits.validation_reserve_s
  return Math.max(1, Math.floor((state.limits.max_elapsed_s - reserve - state.elapsed_s) * 1000))
}

/** What one deterministic node produced, before the plugin judges it. */
interface CommandResult {
  /** The command that ran, for reporting. */
  command: string
  /** The graph node id, for node-level failure recording. */
  nodeId: string
  /** The process exit code; null when the node never ran or died from a signal. */
  exitCode: number | null
  /** Bounded stdout. */
  stdout: string
  /** Bounded stderr. */
  stderr: string
}

/** A node that failed: the reason is always present, so no caller reports a bare failure. */
export type CommandFailure = CommandResult & { ok: false; error: string }

/**
 * The outcome of one deterministic node: it completed, carrying the verdict a
 * gate's evaluator printed and any reason it gave, or it failed with the reason
 * it failed.
 */
export type CommandOutcome =
  | (CommandResult & { ok: true; verdict?: string; reason?: string; missing?: true })
  | CommandFailure

/**
 * The stderr line an optional node (one a fork names in `optional_targets`) prints when
 * it failed and exits 0: `refactory_dsh.run` and the worker print it. The node is
 * recorded failed rather than completed, and its branch goes on with its next node,
 * as the walker records and continues it.
 */
export const OPTIONAL_MISSING_MARKER = '--- refactory optional node missing ---'

/**
 * The most of a gate's reason one payload carries, marker included, so a long
 * check log cannot crowd out the rest of the view. `refactory_dsh.run` clips to
 * the same limit, so what it prints arrives here unchanged.
 */
export const REASON_LIMIT = 4_000

/** The command a node runs itself, or `undefined` when a model owns the node. */
export function commandOf(node: NodeView): string | undefined {
  if (node._type === 'FnNode' || node._type === 'Study') {
    return node.command === undefined || node.command === '' ? undefined : node.command
  }
  if (node._type === 'GateNode' && node.evaluator_type === 'fn') {
    const command = node.evaluator_command
    return command === undefined || command === '' ? undefined : command
  }
  // A dispatched AgentNode carries its dispatch command in the view's `command`
  // field (mapped by spine.ts describe()). This makes it deterministic:
  // drain() runs it, wf_done refuses it, and wf_next never hands it to the CEO.
  if (node._type === 'AgentNode' && node.command !== undefined && node.command !== '') {
    return node.command
  }
  return undefined
}

/** True when the plugin runs this node rather than handing it to the driver. */
export function isDeterministic(node: NodeView): boolean {
  return commandOf(node) !== undefined
}

/**
 * The verdict a gate's evaluator stdout declares: its last non-empty line,
 * lowercased to match how the spine compares edge conditions.
 * @param stdout - the evaluator's captured stdout.
 * @returns the verdict label, or `undefined` when the evaluator printed nothing.
 */
export function verdictOf(stdout: string): string | undefined {
  const lines = stdout.split('\n').map(line => line.trim()).filter(line => line !== '')
  const last = lines[lines.length - 1]
  return last === undefined ? undefined : last.toLowerCase()
}

/**
 * Why a gate's evaluator routed as it did: what it printed before its verdict.
 *
 * The verdict stays the last line the spine routes on, so an evaluator explains
 * itself by printing its reason first. The reason travels to the reloop target's
 * feedback and to the CEO's `ran` step; without it a reloop reads as an empty
 * string and nobody can say why the run went back. A reason longer than
 * {@link REASON_LIMIT} keeps its head and its tail, where a finding list and a
 * test summary sit, and says how long it was.
 * @param stdout - the evaluator's captured stdout.
 * @returns the reason, or an empty string when the evaluator printed only its verdict.
 */
export function reasonOf(stdout: string): string {
  const lines = stdout.split('\n')
  let last = lines.length - 1
  while (last >= 0 && lines[last]!.trim() === '') last -= 1
  const reason = lines.slice(0, Math.max(last, 0)).join('\n').trim()
  if (reason.length <= REASON_LIMIT) return reason
  const marker = `\n… (clipped from ${reason.length} characters) …\n`
  const half = Math.floor((REASON_LIMIT - marker.length) / 2)
  return `${reason.slice(0, half)}${marker}${reason.slice(-half)}`
}

/**
 * A line a node addresses to the person driving the run: `run warning: …`,
 * optionally after the program's name, as in `refactory: run warning: …`. Narrower
 * than any `warning:` on purpose: a routine notice such as `refactory: warning:
 * credential readiness is unverified` stays in the log, because repeating it on
 * every run would teach the person to stop reading warnings.
 */
const RUN_WARNING = /^(?:[\w.-]+:\s*)?run warning:\s*\S/i

/**
 * The run warnings a node that completed printed to stderr, one per line.
 *
 * Nobody is shown a completed node's stderr, so a step that noticed, say, the
 * run's target repository moving under it (#330 I1) would warn nobody. These lines
 * travel with the `ran` step instead, for the CEO to pass on.
 * @param stderr - the node's captured stderr.
 * @returns each run-warning line, trimmed, in the order it was printed.
 */
export function runWarningsOf(stderr: string): string[] {
  return stderr.split('\n').map(line => line.trim()).filter(line => RUN_WARNING.test(line))
}

/** The declaration handed to a node command. */
function manifestOf(node: NodeView, command: string, promptFile?: string, feedbackFile?: string): NodeManifest {
  return {
    id: node.id,
    type: node._type,
    command,
    reads: node.reads ?? [],
    writes: node.writes ?? [],
    ...(promptFile !== undefined ? { prompt_file: promptFile } : {}),
    ...(feedbackFile !== undefined ? { feedback_file: feedbackFile } : {}),
  }
}

/** One command's request: where it runs, with what, and under which sandbox. */
function requestFor(node: NodeView, command: string, config: CommandRunnerConfig, promptFile?: string, feedbackFile?: string): ShellExecRequest {
  const seconds = node.command_timeout_s
  if (seconds !== undefined && (!Number.isSafeInteger(seconds) || seconds <= 0 || seconds > 2_147_483)) {
    throw new Error(`${node.id} has an invalid command_timeout_s`)
  }
  return {
    command,
    env: { ...config.env, [NODE_ENV_KEY]: JSON.stringify(manifestOf(node, command, promptFile, feedbackFile)) },
    timeoutMs: seconds === undefined ? config.timeoutMs : seconds * 1000,
    ...(config.workdir === undefined ? {} : { workdir: config.workdir }),
    ...(config.sandboxPolicy === undefined ? {} : { sandboxPolicy: config.sandboxPolicy }),
  }
}

/** One argument, quoted so a shell hands it over whole. */
function quoted(value: string): string {
  return `'${value.replace(/'/g, String.raw`'\''`)}'`
}

/**
 * Run one resolved spec to its foreground result.
 *
 * The shell seam splits preparation from execution: `execute` returns the live
 * process handle and `result()` is the foreground projection that settles with
 * the split streams and the first-cause timeout/abort classification. A graph
 * node only ever wants that projection, so every deterministic node runs
 * through this one adapter.
 * @param shell - the shell executor that runs node commands.
 * @param spec - a fully resolved spec from `shell.resolve`.
 * @returns the settled foreground result for the command.
 */
async function runSpec(shell: ShellExecutor, spec: ShellExecSpec): Promise<ShellRunResult> {
  const execution = await shell.execute(spec)
  return execution.result()
}

/**
 * The item command a data node runs: one verb of its loader, naming the node.
 *
 * The loader is deployment state (`dataCommand`), not graph structure: the graph
 * says which items the node selected, and the command that reads them is what the
 * composition mounts. The node id travels in the command line because the loader is
 * a separate process with the run directory and no access to the graph.
 * @param node - the data node whose items to act on.
 * @param verb - the loader verb: `list`, `begin`, or `end`.
 * @param config - the data loader command and the settings node commands run under.
 * @param index - the item's position in the selection, for `begin` and `end`.
 * @returns the command line.
 */
export function dataCommandOf(
  node: NodeView,
  verb: string,
  config: { dataCommand: string },
  index?: number,
  failedReason?: string,
  failedKind?: string,
  extra: readonly string[] = [],
): string {
  const position = index === undefined ? '' : ` --index ${String(index)}`
  const failed = failedReason === undefined ? '' : ` --failed ${quoted(failedReason)}`
  const kind = failedKind === undefined ? '' : ` --failed-kind ${quoted(failedKind)}`
  const flags = extra.map(flag => ` ${quoted(flag)}`).join('')
  return `${config.dataCommand} ${verb} --node ${quoted(node.id)}${position}${failed}${kind}${flags}`
}

/**
 * Run one of a data node's item commands and read the JSON it prints.
 *
 * The item loop is three commands because the pass between them can contain agent
 * nodes: the runtime drives the subgraph, so it has to learn the selection, start
 * one pass, and close it, rather than owning the whole loop the way the walking
 * executor does.
 * @param shell - the shell executor that runs node commands.
 * @param node - the data node view.
 * @param verb - the loader verb: `list`, `begin`, `end`, or a concurrent pass's
 * `prepare` and `finish`.
 * @param config - the data loader command and the settings node commands run under.
 * @param index - the item's position, for `begin` and `end`.
 * @param extra - further arguments, such as `--concurrent`, each quoted.
 * @returns the parsed JSON the command printed, or a failure carrying its stderr.
 */
export async function runDataCommand(
  shell: ShellExecutor,
  node: NodeView,
  verb: string,
  config: CommandRunnerConfig & { dataCommand: string },
  index?: number,
  failedReason?: string,
  failedKind?: string,
  extra: readonly string[] = [],
): Promise<{ ok: true; value: Record<string, unknown> } | CommandFailure> {
  const command = dataCommandOf(node, verb, config, index, failedReason, failedKind, extra)
  const allowance = resourceCheck(node, config)
  const resolved = shell.resolve(requestFor(node, command, config))
  const spec = allowance === undefined ? resolved : { ...resolved, timeoutMs: Math.min(resolved.timeoutMs, allowance) }
  const result = await runSpec(shell, spec)
  resourceCheck(node, config)
  const stdout = result.stdout.text.trim()
  const stderr = result.stderr.text.trim()
  const base: CommandResult = { command, nodeId: node.id, exitCode: result.exitCode, stdout, stderr }
  if (result.timedOut) {
    return { ...base, ok: false, error: `${node.id} ${verb} timed out after ${result.timeoutMs}ms` }
  }
  if (result.aborted) return { ...base, ok: false, error: `${node.id} ${verb} was aborted` }
  if (result.exitCode !== 0) {
    return { ...base, ok: false, error: `${node.id} ${verb} exited with code ${String(result.exitCode)}` }
  }
  try {
    const value: unknown = JSON.parse(stdout)
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      return { ...base, ok: false, error: `${node.id} ${verb} printed ${stdout}, not a JSON object` }
    }
    return { ok: true, value: value as Record<string, unknown> }
  } catch {
    // The loader's stdout is its answer, so unparseable output is the failure: a
    // command that printed a usage line exited 0 somewhere it should not have.
    return { ...base, ok: false, error: `${node.id} ${verb} printed ${stdout}, not JSON` }
  }
}

/**
 * Run one deterministic node.
 *
 * A nonzero exit, a timeout, an abort, or (for a gate) empty stdout fails the
 * node rather than advancing the cursor: a graph step that did not happen must
 * not look like one that did.
 *
 * Exit code 3 with the `--- refactory failure ---` marker in stderr is a
 * build-loop limit stop: the spine records `failed` (terminal), not `ready`
 * (retryable).  Every other nonzero exit stays `ready` (dm-95).
 * @param shell - the shell executor that runs node commands.
 * @param node - the node view whose command to run.
 * @param config - work directory, environment, and timeout for the command.
 * @returns the outcome; never rejects for a command's own failure.
 */
export async function runDeterministic(
  shell: ShellExecutor,
  node: NodeView,
  config: CommandRunnerConfig,
  feedbackFile?: string,
): Promise<CommandOutcome> {
  const command = commandOf(node)
  const base: CommandResult = {
    command: command ?? '',
    nodeId: node.id,
    exitCode: null,
    stdout: '',
    stderr: '',
  }
  if (command === undefined) {
    return { ...base, ok: false, error: `${node.id} has no command` }
  }

  // Write the current prompt to a file for dispatched AgentNodes.
  // The prompt_template may have been amended by wf_edit; the spec file
  // holds the original. The prompt_file is what the worker reads.
  let promptFile: string | undefined
  if (node._type === 'AgentNode' && node.prompt_template !== undefined && config.workdir !== undefined) {
    const dir = join(config.workdir, '.refactory', 'prompts')
    mkdirSync(dir, { recursive: true })
    promptFile = join(dir, `${node.id}.md`)
    writeFileSync(promptFile, node.prompt_template)
  }

  const allowance = resourceCheck(node, config)
  const request = requestFor(node, command, config, promptFile, feedbackFile)
  const resolved = shell.resolve(request)
  // The model-facing shell caps arbitrary calls at its default maximum. This
  // trusted workflow operation carries a bounded allowance resolved at lowering;
  // retain every resolved environment/sandbox setting while honoring that budget.
  const timeoutMs = node.command_timeout_s === undefined ? resolved.timeoutMs : node.command_timeout_s * 1000
  const spec = { ...resolved, timeoutMs: allowance === undefined ? timeoutMs : Math.min(timeoutMs, allowance) }
  const result = await runSpec(shell, spec)
  resourceCheck(node, config)
  const stdout = result.stdout.text.trim()
  const stderr = result.stderr.text.trim()
  const outcome: CommandResult = { ...base, exitCode: result.exitCode, stdout, stderr }

  if (result.timedOut) {
    return { ...outcome, ok: false, error: `${node.id} timed out after ${result.timeoutMs}ms` }
  }
  if (result.aborted) return { ...outcome, ok: false, error: `${node.id} was aborted` }
  if (result.exitCode !== 0) {
    return { ...outcome, ok: false, error: `${node.id} exited with code ${String(result.exitCode)}` }
  }

  if (node._type === 'GateNode') {
    const verdict = verdictOf(stdout)
    if (verdict === undefined) return { ...outcome, ok: false, error: `${node.id} printed no verdict` }
    const reason = reasonOf(stdout)
    return { ...outcome, ok: true, verdict, ...(reason === '' ? {} : { reason }) }
  }
  if (stderr.split('\n').some(line => line.trim() === OPTIONAL_MISSING_MARKER)) {
    return { ...outcome, ok: true, missing: true }
  }
  return { ...outcome, ok: true }
}
