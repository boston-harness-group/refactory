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
}

/** Per-run settings for node commands: where they run, and with what. */
export interface CommandRunnerConfig {
  /** Caller-owned cancellation for advisory commands. */
  signal?: AbortSignal
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
  const checked = spawnSync('python', ['-m', 'refactory_dsh.budgets', ledger, '--check', ...(phase === undefined ? [] : ['--phase', phase])], {
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
 * gate's evaluator printed, or it failed with the reason it failed.
 */
export type CommandOutcome =
  | (CommandResult & { ok: true; verdict?: string })
  | CommandFailure

/** The command a node runs itself, or `undefined` when a model owns the node. */
export function commandOf(node: NodeView): string | undefined {
  if (node._type === 'FnNode' || node._type === 'Study') {
    return node.command === undefined || node.command === '' ? undefined : node.command
  }
  if (node._type === 'GateNode' && (node.evaluator_type === 'fn'
      || (node.evaluator_type === 'agent' && node.evaluator_command?.startsWith('python -m refactory_dsh.run refactory_dsh.create_fidelity_gates:evaluate_native')))) {
    const command = node.evaluator_command
    return command === undefined || command === '' ? undefined : command
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

/** The declaration handed to a node command. */
function manifestOf(node: NodeView, command: string): NodeManifest {
  return {
    id: node.id,
    type: node._type,
    command,
    reads: node.reads ?? [],
    writes: node.writes ?? [],
  }
}

/** One command's request: where it runs, with what, and under which sandbox. */
function requestFor(node: NodeView, command: string, config: CommandRunnerConfig): ShellExecRequest {
  const seconds = node.command_timeout_s
  if (seconds !== undefined && (!Number.isSafeInteger(seconds) || seconds <= 0 || seconds > 2_147_483)) {
    throw new Error(`${node.id} has an invalid command_timeout_s`)
  }
  return {
    command,
    env: { ...config.env, [NODE_ENV_KEY]: JSON.stringify(manifestOf(node, command)) },
    timeoutMs: seconds === undefined ? config.timeoutMs : seconds * 1000,
    ...(config.signal === undefined ? {} : { signal: config.signal }),
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
): string {
  const position = index === undefined ? '' : ` --index ${String(index)}`
  return `${config.dataCommand} ${verb} --node ${quoted(node.id)}${position}`
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
 * @param verb - the loader verb: `list`, `begin`, or `end`.
 * @param config - the data loader command and the settings node commands run under.
 * @param index - the item's position, for `begin` and `end`.
 * @returns the parsed JSON the command printed, or a failure carrying its stderr.
 */
export async function runDataCommand(
  shell: ShellExecutor,
  node: NodeView,
  verb: string,
  config: CommandRunnerConfig & { dataCommand: string },
  index?: number,
): Promise<{ ok: true; value: Record<string, unknown> } | CommandFailure> {
  const command = dataCommandOf(node, verb, config, index)
  const allowance = resourceCheck(node, config)
  const resolved = shell.resolve(requestFor(node, command, config))
  const spec = allowance === undefined ? resolved : { ...resolved, timeoutMs: Math.min(resolved.timeoutMs, allowance) }
  const result = await runSpec(shell, spec)
  resourceCheck(node, config)
  const stdout = result.stdout.text.trim()
  const stderr = result.stderr.text.trim()
  const base: CommandResult = { command, exitCode: result.exitCode, stdout, stderr }
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
 * @param shell - the shell executor that runs node commands.
 * @param node - the node view whose command to run.
 * @param config - work directory, environment, and timeout for the command.
 * @returns the outcome; never rejects for a command's own failure.
 */
export async function runDeterministic(
  shell: ShellExecutor,
  node: NodeView,
  config: CommandRunnerConfig,
): Promise<CommandOutcome> {
  const command = commandOf(node)
  const base: CommandResult = {
    command: command ?? '',
    exitCode: null,
    stdout: '',
    stderr: '',
  }
  if (command === undefined) {
    return { ...base, ok: false, error: `${node.id} has no command` }
  }

  const allowance = resourceCheck(node, config)
  const request = requestFor(node, command, config)
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
    return { ...outcome, ok: true, verdict }
  }
  return { ...outcome, ok: true }
}
