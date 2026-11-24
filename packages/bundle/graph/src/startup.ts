/**
 * The graph app's command-line provider: it parses `--mode/--task/--budget`
 * (and `--provider/--model/--graphs/--workdir`), loads the mode's committed
 * `graph.json`, and publishes {@link GRAPH_STARTUP_SERVICE}. The workflow-spine
 * row and the runner both read from that service, so a run is parameterised by
 * the command line rather than baked into a profile.
 * @module @deepseek-ai/dsh-graph/startup
 */

import { readFileSync, existsSync, mkdirSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { Command } from 'commander'
import type { Context } from '@deepseek-ai/cordis'
import { parseCmdline } from '@deepseek-ai/dsh-cmdline'

/** Stable Cordis plugin name. */
export const name = 'graph-startup'

/** Services required before the run can be resolved. */
export const inject = ['cmdlineArgs']

/** Service provided by this plugin and injected by the runner and spine row. */
export const GRAPH_STARTUP_SERVICE = 'graphStartup'

/** The mode's state file, used by the runner to report the best score. */
const STATE_FILES: Record<string, string> = {
  gepa: 'gepa_state.json',
  best_of_n: 'best_of_n_result.json',
  scs: 'scs_state.json',
  aide: 'aide_state.json',
  ai_sci_v1: 'autoresearch_state.json',
  ai_sci_v2: 'aide_state.json',
  openevolve: 'openevolve_state.json',
  shinka: 'shinka_state.json',
  adaevolve: 'adaevolve_state.json',
  evox: 'evox_state.json',
  autoresearch: 'autoresearch_state.json',
  karpathy: 'karpathy_state.json',
  autoscientists: 'autoscientists_state.json',
}

/** Resolve a dependency path from an explicit env override, else relative to
 * `REFACTORY_DSH_HOME` (the repo root), else relative to the checkout the CLI
 * runs from. Keeps the bundle self-contained instead of baking an
 * absolute machine path: a fresh clone's `.venv` and vendored SRF graphs are
 * found without the pre-consolidation `~/dsh/refactory-dsh` layout. */
function resolveDep(override: string | undefined, homeRelative: string, fallback: string): string {
  if (override) return override
  const home = process.env.REFACTORY_DSH_HOME
  if (home) return `${home}/${homeRelative}`
  return fallback
}

const DEFAULT_GRAPHS_DIR = resolveDep(process.env.SRF_GRAPHS, 'refactory/vendor/srf/graphs', join(process.cwd(), 'refactory/vendor/srf/graphs'))
const DEFAULT_VENV_BIN = resolveDep(process.env.VENV_BIN, '.venv/bin', join(process.cwd(), '.venv/bin'))


/** Everything the run needs, resolved once from the command line. */
export interface GraphStartupValues {
  /** The CEO driving prompt. */
  task: string
  /** The mode's graph, exactly refactory's `Workflow.to_dict()` output. */
  graph: unknown
  /** Working directory for node commands and the ops' artifacts. */
  workdir: string
  /** Environment entries for node commands (SRF_* plus the venv on PATH). */
  commandEnv: Record<string, string>
  /** The mode name, for reporting and state-file lookup. */
  mode: string
  /** The task name, passed to the ops as SRF_TASK. */
  taskName: string
  /** The eval budget, passed to the ops as SRF_BUDGET. */
  budget: number
  /** The CEO model route, for the runner's agent creation; a CEO launch without one is refused (dm-258). */
  provider: string | undefined
  /** The CEO model id; required with the provider, never taken from the profile's default. */
  model: string | undefined
  /** The directory of the Python that has refactory_dsh installed. */
  pythonBin: string
  /** The command that selects a data node's items and opens and closes one pass. */
  dataCommand: string
  /** Shared artifact delivery/provenance bridge for agent nodes. */
  agentCommand: string
  /** Shared review-scoped user-gate ledger bridge. */
  gateCommand: string
  /** The steer command executable, or '' when no refactory.steer entry point is installed. */
  steerCommand?: string
  /** The steer command flags, or [] when no refactory.steer entry point is installed. */
  steerArgs?: string[]
  /**
   * Whether the runner keeps a conversation with the person after the first turn.
   *
   * Off is the one-shot run scripts and CI get: send the driving task, report the
   * result, exit. On, the runner prints each turn and reads a line between turns,
   * so a person can steer the CEO (`wf_edit` + `wf_redo`, skip, route) while the
   * graph is still running. `ceo` asks for it unless `--headless` was given.
   */
  chat: boolean
}

function graphCommand(): Command {
  return new Command()
    .name('dsh --profile graph')
    .description('Drive a refactory mode graph through the spine, with one CEO Agent.')
    .helpOption('-h, --help', 'show this help')
    .argument('[command]', 'the subcommand; "run" drives the graph to completion', 'run')
    .requiredOption('-m, --mode <mode>', 'the SRF mode (e.g. gepa, best_of_n)')
    .requiredOption('-t, --task <task>', 'the SRF task (e.g. circle_packing)')
    .option('-b, --budget <n>', 'eval budget', '10')
    .option('-p, --provider <provider>', 'provider route for the CEO model (refactory passes the run\'s route; a CEO run without one is refused)')
    .option('--model <model>', 'model id for the CEO (required with --provider; never the profile default)')
    .option('--graphs <dir>', 'directory of <mode>.graph.json', DEFAULT_GRAPHS_DIR)
    .option('--workdir <dir>', 'working directory for the run')
    .option('--python-bin <dir>', 'directory of the python with srf+factory installed', DEFAULT_VENV_BIN)
    .option('--knobs <json>', 'JSON object of OptKnob overrides (e.g. {"temperature":0.5})', '{}')
    .option(
      '--chat',
      'keep the session live: print each turn and read what the person types between '
      + 'turns, until /quit, end of input, or the graph reports done (default: one-shot)',
      false,
    )
    .addHelpText('after', `
Examples:
  dsh --profile graph run --mode gepa --task circle_packing --budget 20
  dsh --profile graph run --mode gepa --task circle_packing --chat
`)
}

/** The mode's graph.json, or a thrown error naming the fix. */
function loadGraph(mode: string, graphsDir: string): unknown {
  const path = join(graphsDir, `${mode}.graph.json`)
  if (!existsSync(path)) {
    throw new Error(`no graph at ${path}; run \`srf graphs --out ${graphsDir}\` first`)
  }
  return JSON.parse(readFileSync(path, 'utf8'))
}

/** Resolve the project path: from the environment or from the process's cwd. */
export function resolveRefactoryProject(): string {
  return process.env.REFACTORY_PROJECT || process.cwd()
}

/**
 * Parse and provide the run as an ordinary Cordis service. `--help` and usage
 * errors provide nothing, which fails the runner row (and therefore boot) loudly.
 * @param ctx - plugin context carrying the command line.
 */
export function apply(ctx: Context): void {
  const program = graphCommand()
  program.action(() => {
    const options = program.opts()
    const mode = String(options.mode)
    const taskName = String(options.task)
    const budget = Number(options.budget)
    const graphsDir = String(options.graphs)
    const workdir = String(options.workdir ?? join(tmpdir(), `dsh-run-${mode}-${Date.now()}`))
    const pythonBin = String(options.pythonBin)

    // Discover the steer module from the refactory.steer entry point.
    // Guard on existsSync to keep existing tests fast: /venv/bin doesn't exist.
    let steerCommand = ''
    let steerArgs: string[] = []
    const pythonPath = `${pythonBin}/python`
    if (existsSync(pythonPath)) {
      const probe = spawnSync(pythonPath, ['-m', 'refactory_dsh.steer_probe'], {
        encoding: 'utf8', timeout: 5000,
      })
      const steerModule = probe.status === 0 ? probe.stdout.trim() : ''
      if (steerModule) {
        steerCommand = pythonPath
        steerArgs = ['-m', steerModule, 'steer', '--run']
      } else {
        // Report the probe's actual error when it failed (e.g. more-than-one RuntimeError).
        const probeError = probe.status !== 0
          ? (probe.stderr?.trim().split('\n').pop() || 'no refactory.steer entry point found')
          : 'no refactory.steer entry point found'
        console.warn(
          'graphStartup: %s in %s; wf_steer will not be registered',
          probeError,
          pythonBin,
        )
      }
    } else {
      console.warn(
        'graphStartup: python not found at %s; steer probe skipped',
        pythonPath,
      )
    }

    const graph = loadGraph(mode, graphsDir)
    // A node command's cwd must exist before the first one spawns.
    mkdirSync(workdir, { recursive: true })
    // The run's interpreter leads the ambient PATH. A stripped environment has no
    // PATH at all, and emitting a trailing separator there would put the current
    // directory on the search path, so the bare interpreter is the whole value.
    const ambientPath = process.env.PATH ?? ''
    const commandEnv: Record<string, string> = {
      SRF_WORK_DIR: workdir,
      SRF_TASK: taskName,
      SRF_KNOBS: String(options.knobs),
      SRF_BUDGET: String(budget),
      REFACTORY_WORKDIR: workdir,
      ...(process.env.REFACTORY_WORKER_CONTEXT ? { REFACTORY_WORKER_CONTEXT: process.env.REFACTORY_WORKER_CONTEXT } : {}),
      ...(process.env.REFACTORY_BUDGET_LEDGER ? { REFACTORY_BUDGET_LEDGER: process.env.REFACTORY_BUDGET_LEDGER } : {}),
      // A watched run's watch directory (refactory_dsh/activity.py `WATCH_ENV`): each node
      // command marks it as it prints and writes, and records its process group there, so
      // an inactivity limit counts a working node and a stop can end it.
      ...(process.env.REFACTORY_RUN_WATCH ? { REFACTORY_RUN_WATCH: process.env.REFACTORY_RUN_WATCH } : {}),
      // Nonsecret context for nested model calls after command/sandbox scrubbing.
      REFACTORY_MODEL_DSH_HOME: resolve((process.env.DSH_HOME?.trim() || join(homedir(), '.dsh')).replace(/^~(?=[/\\]|$)/, homedir())),
      REFACTORY_MODEL_PROVIDER: String(options.provider ?? ''),
      REFACTORY_MODEL_ID: String(options.model ?? ''),
      // The node commands run *in the work directory*, so `Path.cwd()` inside one of
      // them is the run, not the checkout. The project — where `.factory/packages`
      // and the packages of modes live — has to be said out loud, and this is where
      // the checkout is known: the directory the profile was started in.
      REFACTORY_PROJECT: resolveRefactoryProject(),
      REFACTORY_INTENT: taskName,
      // An op node's command is lowered as a bare `python -m refactory_dsh.run` (lowering.RUNNER),
      // so graphs name no machine path; the run's interpreter leading PATH is what resolves it.
      // Agents get the target's own PATH at dispatch instead (model._environment).
      PATH: ambientPath === '' ? pythonBin : `${pythonBin}:${ambientPath}`,
      REFACTORY_PYTHON_BIN: `${pythonBin}/python`,
    }

    ctx.provide(GRAPH_STARTUP_SERVICE, {
      task: 'Drive the workflow graph subject to the person\'s current steering. BEFORE the first wf_next, '
        + 'read wf_steer when available; its ledger contains the actual user request. Use wf_state for '
        + 'inspection without execution. Arm any requested stop with wf_edit pause on that node before '
        + 'advancing. When paused, yield your turn and wait for the person; only explicit permission to '
        + 'continue authorizes wf_edit unpause. Otherwise call wf_next after every wf_done/wf_verdict. '
        + 'wf_next crosses sole static fork/join boundaries and establishes branch delivery contexts '
        + 'for you. Dispatch only the AgentNode/LLMNode prompts it returns, never fork target ids from '
        + 'a state view. If a control node is explicitly returned, wf_done it, then call wf_next first. '
        + 'For each AgentNode or LLMNode (unless it has a dispatch command — the runtime runs those), spawn a subagent and pass the returned prompt in full. '
        + 'Coordination notes must not weaken its requirements or waive the original brief/approved design '
        + 'without explicit user steering; a documented limitation is not a waiver. Preserve the node\'s '
        + 'declared `writes` contract and read its `reads`, plus any inputs expressly required by its prompt. '
        + 'Perform checks or delegate only as that node expressly requires. Do not independently rerun '
        + 'the eval or deterministic FnNode/gate commands, add unrelated edits, or iterate beyond the node\'s '
        + 'requested work: the deterministic commands run automatically. After writing the required outputs, '
        + 'stop and return a one-line summary. Then report wf_done for the node. Route each '
        + 'gate with wf_verdict. When wf_next returns done, stop and report the final best score you observed.',
      graph,
      workdir,
      commandEnv,
      mode,
      taskName,
      budget,
      provider: options.provider === undefined ? undefined : String(options.provider),
      model: options.model === undefined ? undefined : String(options.model),
      pythonBin,
      dataCommand: `${pythonBin}/python -m refactory_dsh.items`,
      agentCommand: `${pythonBin}/python -m refactory_dsh.agent_delivery`,
      steerCommand,
      steerArgs,
      gateCommand: `${pythonBin}/python -m refactory_dsh.steering`,
      chat: options.chat === true,
    } satisfies GraphStartupValues)
  })
  parseCmdline(ctx, program)
}

export { STATE_FILES }
